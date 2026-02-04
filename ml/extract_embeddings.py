import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
from tf_require import require_tensorflow
from PIL import Image
from tqdm import tqdm

tf = require_tensorflow()


def _load_paths_from_directory(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths: List[str] = []
    labels: List[int] = []
    for c in class_names:
        root = os.path.join(data_dir, c)
        for name in os.listdir(root):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            paths.append(os.path.join(root, name))
            labels.append(class_to_idx[c])

    return paths, labels, class_names


def _load_csv(csv_path: str, image_root: str, path_col: str, label_col: Optional[str]) -> Tuple[List[str], Optional[List[str]]]:
    import csv

    paths: List[str] = []
    labels: Optional[List[str]] = [] if label_col else None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = os.path.join(image_root, row[path_col])
            if not os.path.exists(p):
                continue
            paths.append(p)
            if label_col:
                assert labels is not None
                labels.append(row.get(label_col, ""))

    return paths, labels


def _load_image(path: str, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to embedding .keras (hand_embedding.keras)")
    ap.add_argument("--mode", choices=["directory", "csv"], required=True)
    ap.add_argument("--data_dir", help="Root directory (mode=directory)")
    ap.add_argument("--csv", dest="csv_path", help="CSV file (mode=csv)")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--path_col", default="path")
    ap.add_argument("--label_col", default="", help="Optional label column name")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True, help="Output .npz")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)

    if args.mode == "directory":
        if not args.data_dir:
            raise SystemExit("--data_dir is required for mode=directory")
        paths, y_int, class_names = _load_paths_from_directory(args.data_dir)
        y = np.array(y_int, dtype=np.int32)
        y_text = np.array([class_names[i] for i in y_int], dtype=object)
    else:
        if not args.csv_path:
            raise SystemExit("--csv is required for mode=csv")
        paths, labels = _load_csv(args.csv_path, args.image_root, args.path_col, args.label_col or None)
        y = None
        y_text = np.array(labels, dtype=object) if labels is not None else None

    if not paths:
        raise SystemExit("No images found")

    X_list: List[np.ndarray] = []
    for i in tqdm(range(0, len(paths), args.batch), desc="Embedding"):
        batch_paths = paths[i : i + args.batch]
        batch = np.stack([_load_image(p, args.img) for p in batch_paths], axis=0)
        emb = model.predict(batch, verbose=0)
        X_list.append(emb.astype(np.float32))

    X = np.concatenate(X_list, axis=0)

    out = {
        "X": X,
        "paths": np.array(paths, dtype=object),
    }
    if y is not None:
        out["y"] = y
        out["y_text"] = y_text
    elif y_text is not None:
        out["y_text"] = y_text

    np.savez(args.out, **out)
    print("Wrote:", args.out)
    print("X shape:", X.shape)


if __name__ == "__main__":
    main()
