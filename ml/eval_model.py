import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tf_require import require_tensorflow

tf = require_tensorflow()


def _maybe_import_sklearn_metrics():
    try:
        from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

        return classification_report, confusion_matrix
    except Exception:
        return None, None


def _extract_probs(pred) -> np.ndarray:
    # Our training script uses a dict output with key "class_output".
    if isinstance(pred, dict):
        pred = pred.get("class_output", pred)
    return np.asarray(pred)


def _load_class_names_from_model_dir(model_path: str) -> Optional[List[str]]:
    model_dir = os.path.dirname(os.path.abspath(model_path))
    p = os.path.join(model_dir, "class_names.txt")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return names or None
    except Exception:
        return None


def _resolve_eval_dir(data_root: str, subset: str) -> str:
    # If data_root has a split layout, default to test.
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    test_root = os.path.join(data_root, "test")

    has_split = os.path.isdir(train_root) and os.path.isdir(val_root)
    if subset == "auto":
        if has_split and os.path.isdir(test_root):
            return test_root
        return data_root

    if subset == "train":
        if not os.path.isdir(train_root):
            raise SystemExit(f"No train/ folder found under: {data_root}")
        return train_root
    if subset == "val":
        if not os.path.isdir(val_root):
            raise SystemExit(f"No val/ folder found under: {data_root}")
        return val_root
    if subset == "test":
        if not os.path.isdir(test_root):
            raise SystemExit(f"No test/ folder found under: {data_root}")
        return test_root

    # subset == root
    return data_root


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    # Macro precision/recall/F1 without sklearn.
    eps = 1e-12
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2.0 * prec * rec / (prec + rec + eps)

        precision_sum += prec
        recall_sum += rec
        f1_sum += f1

    acc = float(np.mean(y_true == y_pred))
    return {
        "accuracy": acc,
        "precision_macro": precision_sum / float(max(1, num_classes)),
        "recall_macro": recall_sum / float(max(1, num_classes)),
        "f1_macro": f1_sum / float(max(1, num_classes)),
    }


def evaluate_model(
    model_path: str,
    data_root: str,
    subset: str = "auto",
    img: int = 224,
    batch: int = 32,
    seed: int = 1337,
    quiet: bool = False,
) -> Tuple[Dict[str, float], Optional[np.ndarray], List[str]]:
    """Evaluate a model on a folder dataset.

    Returns:
      - metrics dict (accuracy, macro precision/recall/F1)
      - confusion matrix if sklearn is available else None
      - class_names list
    """
    eval_dir = _resolve_eval_dir(data_root, subset)
    if not quiet:
        print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    image_size: Tuple[int, int] = (img, img)
    class_names_from_file = _load_class_names_from_model_dir(model_path)

    if not quiet:
        print(f"Preparing dataset from: {eval_dir}")
    ds = tf.keras.utils.image_dataset_from_directory(
        eval_dir,
        label_mode="int",
        image_size=image_size,
        batch_size=batch,
        shuffle=False,
        seed=seed,
        class_names=class_names_from_file,
    )
    class_names = list(ds.class_names)
    num_classes = len(class_names)

    y_true_list: List[int] = []
    y_pred_list: List[int] = []

    if not quiet:
        print("Running predictions...")
    for batch_x, batch_y in ds:
        probs = _extract_probs(model.predict(batch_x, verbose=0))
        y_pred = np.argmax(probs, axis=1).astype(np.int32)
        y_true = batch_y.numpy().astype(np.int32)
        y_true_list.extend(y_true.tolist())
        y_pred_list.extend(y_pred.tolist())

    y_true_arr = np.asarray(y_true_list, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred_list, dtype=np.int32)

    classification_report, confusion_matrix = _maybe_import_sklearn_metrics()
    cm = confusion_matrix(y_true_arr, y_pred_arr) if confusion_matrix is not None else None

    metrics = _compute_basic_metrics(y_true_arr, y_pred_arr, num_classes=num_classes)

    if not quiet and classification_report is not None:
        print("\nClassification report:")
        print(classification_report(y_true_arr, y_pred_arr, target_names=class_names, digits=4))
        if cm is not None:
            print("\nConfusion matrix (rows=true, cols=pred):")
            print(cm)

        print("\nMetrics:")
        print(f"- accuracy: {metrics['accuracy']:.4f}")
        print(f"- precision_macro: {metrics['precision_macro']:.4f}")
        print(f"- recall_macro: {metrics['recall_macro']:.4f}")
        print(f"- f1_macro: {metrics['f1_macro']:.4f}")

    return metrics, cm, class_names


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a saved Keras model on a folder dataset (no retraining).")
    ap.add_argument("--model", required=True, help="Path to saved model (.keras or .h5)")
    ap.add_argument(
        "--data",
        required=True,
        help="Dataset directory. Can be root with class folders OR a split root with train/val/test/.",
    )
    ap.add_argument("--subset", choices=["auto", "root", "train", "val", "test"], default="auto")
    ap.add_argument("--img", type=int, default=224, help="Image size (square)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    classification_report, _confusion_matrix = _maybe_import_sklearn_metrics()
    if classification_report is None:
        print("(sklearn not available; will still print basic metrics)")

    metrics, cm, class_names = evaluate_model(
        model_path=args.model,
        data_root=args.data,
        subset=args.subset,
        img=args.img,
        batch=args.batch,
        seed=args.seed,
        quiet=False,
    )

    # If sklearn wasn't available, print metrics here.
    if classification_report is None:
        print("\nMetrics:")
        print(f"- accuracy: {metrics['accuracy']:.4f}")
        print(f"- precision_macro: {metrics['precision_macro']:.4f}")
        print(f"- recall_macro: {metrics['recall_macro']:.4f}")
        print(f"- f1_macro: {metrics['f1_macro']:.4f}")

        if cm is not None:
            print("\nConfusion matrix (rows=true, cols=pred):")
            print(cm)


if __name__ == "__main__":
    main()
