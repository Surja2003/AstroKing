from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from tf_require import require_tensorflow

if TYPE_CHECKING:
    import tensorflow as tft  # pyright: ignore[reportMissingImports]

tf = require_tensorflow()


def _maybe_import_sklearn_metrics():
    try:
        from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

        return classification_report, confusion_matrix
    except Exception:
        return None, None


@dataclass
class CsvConfig:
    csv_path: str
    image_root: str
    path_col: str = "path"
    gender_col: str = "gender"
    age_col: str = "age"


def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _build_backbone(input_shape: Tuple[int, int, int]) -> tft.keras.Model:
    # Mobile-friendly backbone.
    return tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )


def _build_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    multitask_age: bool,
    embedding_dim: int,
    dropout: float,
) -> Tuple[tft.keras.Model, tft.keras.Model, tft.keras.Model, dict, dict]:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    backbone = _build_backbone(input_shape)
    backbone._name = "backbone"
    backbone.trainable = False

    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    embedding = tf.keras.layers.Dense(embedding_dim, activation="relu", name="embedding")(x)

    # Primary head: classification (e.g., gender)
    cls_out = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(embedding)

    outputs = {"class_output": cls_out}
    losses = {"class_output": "sparse_categorical_crossentropy"}
    metrics = {"class_output": ["accuracy"]}

    if multitask_age:
        age_out = tf.keras.layers.Dense(1, activation="linear", name="age_output")(embedding)
        outputs["age_output"] = age_out
        losses["age_output"] = "mse"
        metrics["age_output"] = ["mae"]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses, metrics=metrics)

    embedding_model = tf.keras.Model(inputs=inputs, outputs=embedding, name="hand_embedding")
    return model, embedding_model, backbone, losses, metrics


def _extract_y_true(batch_y):
    # Directory datasets return int labels; CSV datasets return dict with class_output.
    if isinstance(batch_y, dict):
        return batch_y["class_output"]
    return batch_y


def _predict_class_probs(model: tft.keras.Model, batch_x: np.ndarray) -> np.ndarray:
    pred = model.predict(batch_x, verbose=0)
    if isinstance(pred, dict):
        pred = pred["class_output"]
    probs = np.asarray(pred)
    return probs


def _find_backbone_layer(model: tft.keras.Model) -> tft.keras.layers.Layer:
    # Older saved models may not have the backbone explicitly named.
    # We heuristically locate a nested model/layer that looks like MobileNetV3.
    candidates: List[tft.keras.layers.Layer] = []
    for layer in model.layers:
        name = getattr(layer, "name", "").lower()
        if "mobilenet" in name:
            candidates.append(layer)

    # Prefer nested models.
    for layer in candidates:
        if isinstance(layer, tf.keras.Model):
            return layer

    if candidates:
        return candidates[0]

    # Fallback: first nested model.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    raise ValueError("No backbone-like layer found")


def _report_confusion(model: tft.keras.Model, ds: tft.data.Dataset, class_names: List[str]) -> None:
    classification_report, confusion_matrix = _maybe_import_sklearn_metrics()
    if classification_report is None or confusion_matrix is None:
        print("\n(sklearn not available; skipping confusion matrix/report)")
        return

    y_true_list: List[int] = []
    y_pred_list: List[int] = []

    for batch_x, batch_y in ds:
        y_true = _extract_y_true(batch_y).numpy().astype(np.int32)
        probs = _predict_class_probs(model, batch_x.numpy())
        y_pred = np.argmax(probs, axis=1).astype(np.int32)
        y_true_list.extend(y_true.tolist())
        y_pred_list.extend(y_pred.tolist())

    y_true_arr = np.array(y_true_list, dtype=np.int32)
    y_pred_arr = np.array(y_pred_list, dtype=np.int32)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true_arr, y_pred_arr))
    print("\nClassification report:")
    print(classification_report(y_true_arr, y_pred_arr, target_names=class_names, digits=4))


def _compute_class_counts(ds: tft.data.Dataset, num_classes: int) -> np.ndarray:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for _x, y in ds:
        y_true = _extract_y_true(y)
        y_np = y_true.numpy().astype(np.int64).reshape(-1)
        for cls in y_np:
            if 0 <= cls < num_classes:
                counts[int(cls)] += 1
    return counts


def _balanced_class_weights(counts: np.ndarray) -> dict:
    # Equivalent to sklearn's "balanced": n_samples / (n_classes * n_i)
    total = int(np.sum(counts))
    n_classes = int(counts.shape[0])
    weights = {}
    for i in range(n_classes):
        c = int(counts[i])
        if c <= 0:
            weights[i] = 0.0
        else:
            weights[i] = total / float(n_classes * c)
    return weights


def _dir_dataset(
    data_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
) -> Tuple[tft.data.Dataset, tft.data.Dataset, Optional[tft.data.Dataset], List[str]]:
    # Supports either:
    # - data_dir/<class>/... (uses validation_split)
    # - data_dir/train/<class>/..., data_dir/val/<class>/..., data_dir/test/<class>/...
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")
    test_root = os.path.join(data_dir, "test")

    if os.path.isdir(train_root) and os.path.isdir(val_root):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_root,
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
        )
        class_names = train_ds.class_names
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_root,
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            class_names=class_names,
        )
        test_ds: Optional[tft.data.Dataset] = None
        if os.path.isdir(test_root):
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_root,
                label_mode="int",
                image_size=image_size,
                batch_size=batch_size,
                seed=seed,
                class_names=class_names,
            )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            validation_split=val_split,
            subset="training",
            seed=seed,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            validation_split=val_split,
            subset="validation",
            seed=seed,
        )
        class_names = train_ds.class_names
        test_ds = None

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def _load_csv_rows(csv_path: str) -> List[dict]:
    import csv

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _csv_dataset(
    cfg: CsvConfig,
    image_size: Tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
    gender_map: Optional[str],
) -> Tuple[tft.data.Dataset, tft.data.Dataset, List[str], bool]:
    rows = _load_csv_rows(cfg.csv_path)
    if not rows:
        raise ValueError("CSV is empty")

    paths: List[str] = []
    genders: List[int] = []
    ages: List[float] = []

    # Gender mapping strategy:
    # - If gender values are strings (male/female), map using --gender_map.
    # - If numeric, parse int directly.
    class_names: List[str]
    if gender_map:
        parts = [p.strip() for p in gender_map.split(",") if p.strip()]
        class_names = parts
        lookup = {name: idx for idx, name in enumerate(class_names)}
    else:
        class_names = []
        lookup = {}

    has_age = cfg.age_col in rows[0]

    for row in rows:
        rel = row[cfg.path_col]
        p = os.path.join(cfg.image_root, rel)
        if not os.path.exists(p):
            # Skip missing paths quietly; you can tighten this if desired.
            continue

        g_raw = row.get(cfg.gender_col, "")
        if g_raw == "":
            continue

        try:
            g = int(g_raw)
        except ValueError:
            if not lookup:
                raise ValueError(
                    "CSV gender column appears non-numeric. Provide --gender_map like 'male,female' "
                    "(matching your CSV values)."
                )
            if g_raw not in lookup:
                raise ValueError(f"Unknown gender label '{g_raw}' not in --gender_map")
            g = lookup[g_raw]

        paths.append(p)
        genders.append(g)

        if has_age:
            try:
                ages.append(float(row[cfg.age_col]))
            except Exception:
                ages.append(float("nan"))

    if not paths:
        raise ValueError("No valid samples found from CSV")

    # If class_names not provided, infer from numeric labels.
    if not class_names:
        unique = sorted(set(genders))
        class_names = [str(u) for u in unique]

    # Basic deterministic split.
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    split = int(len(idx) * (1.0 - val_split))
    train_idx = idx[:split]
    val_idx = idx[split:]

    def make_ds(indices: np.ndarray) -> tft.data.Dataset:
        p = np.array(paths, dtype=object)[indices]
        g = np.array(genders, dtype=np.int32)[indices]

        def load_image(path: tft.Tensor) -> tft.Tensor:
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, image_size)
            return img

        x = tf.data.Dataset.from_tensor_slices(p).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        y = tf.data.Dataset.from_tensor_slices({"class_output": g})

        if has_age:
            a = np.array(ages, dtype=np.float32)[indices]
            y = tf.data.Dataset.from_tensor_slices({"class_output": g, "age_output": a})

        ds = tf.data.Dataset.zip((x, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return make_ds(train_idx), make_ds(val_idx), class_names, has_age


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["directory", "csv"], required=True)

    ap.add_argument("--data_dir", help="Directory dataset root (mode=directory)")
    ap.add_argument("--data", help="Alias for --data_dir")
    ap.add_argument("--csv", dest="csv_path", help="CSV labels file (mode=csv)")
    ap.add_argument("--image_root", default=".", help="Root prefix for CSV paths (mode=csv)")
    ap.add_argument("--path_col", default="path")
    ap.add_argument("--gender_col", default="gender")
    ap.add_argument("--age_col", default="age")
    ap.add_argument("--gender_map", default="", help="Comma-separated class names for string labels, e.g. 'male,female'")

    ap.add_argument("--img", default=224, type=int)
    ap.add_argument("--batch", default=32, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--val_split", default=0.2, type=float)
    ap.add_argument("--seed", default=1337, type=int)
    ap.add_argument("--embedding_dim", default=128, type=int)
    ap.add_argument("--dropout", default=0.2, type=float)
    ap.add_argument("--out_dir", default="models")

    ap.add_argument(
        "--fit_verbose",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Keras fit verbosity: 0=silent, 1=progress bar, 2=one line per epoch (recommended).",
    )

    ap.add_argument("--resume_model", default="", help="Load an existing .keras model instead of training stage 1")

    ap.add_argument("--fine_tune", action="store_true", help="Enable stage-2 fine-tuning of the backbone")
    ap.add_argument("--fine_tune_epochs", type=int, default=5)
    ap.add_argument("--fine_tune_lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze_last", type=int, default=20, help="How many backbone layers to unfreeze")

    ap.add_argument(
        "--class_weight",
        choices=["none", "balanced"],
        default="none",
        help="Apply class weighting to mitigate imbalance (directory/csv). 'balanced' often fixes majority-class collapse.",
    )
    args = ap.parse_args()

    _set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    image_size = (args.img, args.img)
    input_shape = (args.img, args.img, 3)

    # Some people naturally type --data; support both.
    if not args.data_dir and args.data:
        args.data_dir = args.data

    if args.mode == "directory":
        if not args.data_dir:
            raise SystemExit("--data_dir is required for mode=directory")
        train_ds, val_ds, test_ds, class_names = _dir_dataset(
            args.data_dir, image_size=image_size, batch_size=args.batch, val_split=args.val_split, seed=args.seed
        )
        num_classes = len(class_names)
        multitask_age = False
    else:
        if not args.csv_path:
            raise SystemExit("--csv is required for mode=csv")
        cfg = CsvConfig(
            csv_path=args.csv_path,
            image_root=args.image_root,
            path_col=args.path_col,
            gender_col=args.gender_col,
            age_col=args.age_col,
        )
        train_ds, val_ds, class_names, multitask_age = _csv_dataset(
            cfg,
            image_size=image_size,
            batch_size=args.batch,
            val_split=args.val_split,
            seed=args.seed,
            gender_map=args.gender_map or None,
        )
        num_classes = len(class_names)

    losses: dict
    metrics: dict

    if args.resume_model:
        model = tf.keras.models.load_model(args.resume_model)
        try:
            backbone = model.get_layer("backbone")
        except Exception:
            try:
                backbone = _find_backbone_layer(model)
            except Exception:
                raise SystemExit("Could not find a backbone layer in the resumed model")

        try:
            emb_layer = model.get_layer("embedding")
            embedding_model = tf.keras.Model(inputs=model.input, outputs=emb_layer.output, name="hand_embedding")
        except Exception:
            # If layer names changed, we can still proceed without re-saving embedding.
            embedding_model = tf.keras.Model(inputs=model.input, outputs=model.outputs[0], name="hand_embedding")

        # Reconstruct compile config from current task settings.
        losses = {"class_output": "sparse_categorical_crossentropy"}
        metrics = {"class_output": ["accuracy"]}
        if multitask_age:
            losses["age_output"] = "mse"
            metrics["age_output"] = ["mae"]

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses, metrics=metrics)
    else:
        model, embedding_model, backbone, losses, metrics = _build_model(
            input_shape=input_shape,
            num_classes=num_classes,
            multitask_age=multitask_age,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
        )

    aug = tf.keras.Sequential(
        (lambda: [
            tf.keras.layers.RandomFlip("horizontal"),
            # ~20 degrees max rotation.
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.15),
            *( [tf.keras.layers.RandomBrightness(0.2)] if hasattr(tf.keras.layers, "RandomBrightness") else [] ),
        ])(),
        name="augmentation",
    )

    def add_aug(x, y):
        return aug(x, training=True), y

    train_aug = train_ds.map(add_aug, num_parallel_calls=tf.data.AUTOTUNE)

    val_for_fit: tft.data.Dataset = val_ds

    if args.class_weight != "none":
        try:
            counts = _compute_class_counts(train_ds, num_classes=num_classes)
            print("\nTrain class counts:")
            for i, name in enumerate(class_names):
                print(f"- {i} ({name}): {int(counts[i])}")

            weights = _balanced_class_weights(counts)
            print("Train class weights:")
            for i, name in enumerate(class_names):
                print(f"- {i} ({name}): {weights[i]:.4f}")

            weights_vec = tf.constant([weights[i] for i in range(num_classes)], dtype=tf.float32)

            def add_sample_weight(x, y):
                y_true = _extract_y_true(y)
                sw = tf.gather(weights_vec, tf.cast(y_true, tf.int32))

                # For multi-output models, provide per-output weights.
                if isinstance(y, dict):
                    out = {"class_output": sw}
                    if "age_output" in y:
                        out["age_output"] = tf.ones_like(sw)
                    return x, y, out

                return x, y, sw

            train_aug = train_aug.map(add_sample_weight, num_parallel_calls=tf.data.AUTOTUNE)

            # Use weighted validation during training so EarlyStopping/Checkpoint aren't
            # implicitly optimized for the majority class on an imbalanced val split.
            val_for_fit = val_ds.map(add_sample_weight, num_parallel_calls=tf.data.AUTOTUNE)
        except Exception as e:
            print(f"Class-weight computation failed; continuing without sample weights: {e}")

    if args.epochs > 0 and not args.resume_model:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, "hand_cnn.keras"), save_best_only=True),
        ]

        model.fit(
            train_aug,
            validation_data=val_for_fit,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=args.fit_verbose,
        )

    if args.fine_tune:
        # Stage 2: unfreeze a tail of the backbone with low LR.
        backbone.trainable = True
        if args.unfreeze_last > 0:
            for layer in backbone.layers[:-args.unfreeze_last]:
                layer.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(args.fine_tune_lr), loss=losses, metrics=metrics)

        print(
            f"\nFine-tuning enabled: unfreeze_last={args.unfreeze_last}, lr={args.fine_tune_lr}, epochs={args.fine_tune_epochs}"
        )
        ft_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, "hand_cnn.keras"), save_best_only=True),
        ]
        model.fit(
            train_aug,
            validation_data=val_for_fit,
            epochs=args.fine_tune_epochs,
            verbose=args.fit_verbose,
            callbacks=ft_callbacks,
        )

    # Report final metrics explicitly (useful when logs are noisy/truncated).
    try:
        train_metrics = model.evaluate(train_ds, verbose=0, return_dict=True)
        val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
        print("\nFinal train metrics:")
        for k in sorted(train_metrics.keys()):
            print(f"- {k}: {train_metrics[k]:.4f}")
        print("Final val metrics:")
        for k in sorted(val_metrics.keys()):
            print(f"- {k}: {val_metrics[k]:.4f}")
    except Exception as e:
        print(f"Final metric evaluation failed: {e}")

    # Optional test evaluation for pre-split datasets.
    if args.mode == "directory" and test_ds is not None:
        print("Test set evaluation:")
        try:
            test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
            for k in sorted(test_metrics.keys()):
                v = test_metrics[k]
                if isinstance(v, (float, int)):
                    print(f"- {k}: {v:.4f}")
                else:
                    print(f"- {k}: {v}")
        except Exception:
            model.evaluate(test_ds, verbose=0)

        # Research-style diagnostics.
        try:
            _report_confusion(model, test_ds, class_names)
        except Exception as e:
            print(f"Confusion/report failed: {e}")

    # Save final models.
    model.save(os.path.join(args.out_dir, "hand_cnn.keras"))
    try:
        embedding_model.save(os.path.join(args.out_dir, "hand_embedding.keras"))
    except Exception as e:
        print(f"Embedding model save failed: {e}")

    # Save class names for later use.
    with open(os.path.join(args.out_dir, "class_names.txt"), "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print("Saved:")
    print("-", os.path.join(args.out_dir, "hand_cnn.keras"))
    print("-", os.path.join(args.out_dir, "hand_embedding.keras"))


if __name__ == "__main__":
    main()
