import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def _iter_images(root: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        paths.append(p)
    return sorted(paths)


def _load_image(path: str, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _load_meta(meta_path: str) -> Dict[str, dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("meta JSON must be an object mapping label -> {traits, summary}")
    return data


def _embed_paths(model, paths: List[str], img_size: int, batch: int) -> np.ndarray:
    X_list: List[np.ndarray] = []
    for i in tqdm(range(0, len(paths), batch), desc="Embedding"):
        batch_paths = paths[i : i + batch]
        batch_arr = np.stack([_load_image(p, img_size) for p in batch_paths], axis=0)
        emb = model.predict(batch_arr, verbose=0)
        X_list.append(np.asarray(emb, dtype=np.float32))
    return np.concatenate(X_list, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a small archetype (personality) embedding index for backend retrieval.")
    ap.add_argument("--model", required=True, help="Path to embedding .keras (e.g., ml/models/hand_embedding.keras)")
    ap.add_argument("--data_dir", required=True, help="Directory with subfolders per archetype label")
    ap.add_argument("--meta", default="", help="Optional JSON mapping label -> {traits: [...], summary: '...'}")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reduce", choices=["mean", "medoid"], default="mean")
    ap.add_argument("--out", required=True, help="Output .npz (copy into backend as palm_trait_index.npz)")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"Not a directory: {args.data_dir}")

    meta: Dict[str, dict] = {}
    if args.meta:
        meta = _load_meta(args.meta)

    from tf_require import require_tensorflow

    tf = require_tensorflow()
    model = tf.keras.models.load_model(args.model)

    labels: List[str] = []
    prototypes: List[np.ndarray] = []
    summaries: List[Optional[str]] = []
    traits_json: List[Optional[str]] = []

    subdirs = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    if not subdirs:
        raise SystemExit("No subfolders found in --data_dir")

    for label in subdirs:
        folder = os.path.join(args.data_dir, label)
        paths = _iter_images(folder)
        if not paths:
            continue

        X = _embed_paths(model, paths, args.img, args.batch)  # (N, D)

        if args.reduce == "mean":
            proto = X.mean(axis=0)
        else:
            # Medoid: pick the sample closest to the mean (more robust than mean on small sets)
            mean = X.mean(axis=0)
            dists = np.sum((X - mean) ** 2, axis=1)
            proto = X[int(np.argmin(dists))]

        labels.append(label)
        prototypes.append(proto.astype(np.float32))

        m = meta.get(label, {}) if meta else {}
        summary = m.get("summary") if isinstance(m, dict) else None
        traits = m.get("traits") if isinstance(m, dict) else None

        summaries.append(str(summary) if summary is not None else None)
        if isinstance(traits, list):
            traits_json.append(json.dumps([str(t) for t in traits], ensure_ascii=False))
        else:
            traits_json.append(None)

    if not labels:
        raise SystemExit("No images found under --data_dir subfolders")

    Xp = np.stack(prototypes, axis=0).astype(np.float32)

    np.savez(
        args.out,
        X=Xp,
        labels=np.array(labels, dtype=object),
        summaries=np.array(summaries, dtype=object),
        traits_json=np.array(traits_json, dtype=object),
        metric="cosine",
        img=int(args.img),
        reduce=str(args.reduce),
        version="1",
    )

    print("Wrote:", args.out)
    print("Prototypes:", Xp.shape)
    print("Labels:", len(labels))


if __name__ == "__main__":
    main()
