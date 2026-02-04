import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from tf_require import require_tensorflow


EXIT_OK = 0
EXIT_BAD_IMAGE = 2
EXIT_MODEL_LOAD = 3


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_and_preprocess(img_path: str, img_size: int) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _load_class_names(model_path: str) -> Optional[List[str]]:
    # train_cnn.py writes class_names.txt next to the model.
    model_dir = os.path.dirname(os.path.abspath(model_path))
    p = os.path.join(model_dir, "class_names.txt")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names or None


def _extract_class_probs(pred) -> np.ndarray:
    # Our classifier can be:
    # - single-output softmax: (1, C)
    # - multi-output dict: {"class_output": (1, C), "age_output": (1, 1)}
    if isinstance(pred, dict):
        if "class_output" not in pred:
            raise ValueError(f"Model returned outputs {list(pred.keys())}, expected 'class_output'")
        pred = pred["class_output"]

    probs = np.asarray(pred)
    if probs.ndim != 2 or probs.shape[0] != 1:
        raise ValueError(f"Unexpected prediction shape: {probs.shape}")

    probs = probs[0]
    if probs.ndim != 1:
        raise ValueError(f"Unexpected prediction vector shape: {probs.shape}")

    return probs


def predict_class(model_path: str, img_path: str, img_size: int) -> dict:
    tf = require_tensorflow()
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {model_path}\n{e}")

    try:
        img = load_and_preprocess(img_path, img_size)
    except Exception as e:
        raise FileNotFoundError(f"Failed to open image: {img_path}\n{e}")

    pred = model.predict(img, verbose=0)
    try:
        probs = _extract_class_probs(pred)
    except ValueError as e:
        raise RuntimeError(
            "Could not extract class probabilities from this model output.\n"
            f"- Model: {model_path}\n"
            "Tip: use --mode embed with the embedding model (hand_embedding.keras), or use the classifier model (hand_cnn.keras) for --mode predict.\n\n"
            f"Details: {e}"
        )

    class_names = _load_class_names(model_path)
    if class_names is None or len(class_names) != len(probs):
        class_names = [f"class_{i}" for i in range(len(probs))]

    best_idx = int(np.argmax(probs))

    items = [
        {"label": str(name), "p": float(p)}
        for name, p in sorted(zip(class_names, probs), key=lambda t: float(t[1]), reverse=True)
    ]

    return {
        "mode": "predict",
        "model": model_path,
        "image": img_path,
        "img_size": int(img_size),
        "top": items[0] if items else None,
        "probs": items,
    }


def extract_embedding(model_path: str, img_path: str, img_size: int, head: int) -> dict:
    tf = require_tensorflow()
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {model_path}\n{e}")

    try:
        img = load_and_preprocess(img_path, img_size)
    except Exception as e:
        raise FileNotFoundError(f"Failed to open image: {img_path}\n{e}")

    emb = model.predict(img, verbose=0)
    emb = np.asarray(emb)

    if emb.ndim == 2 and emb.shape[0] == 1:
        emb = emb[0]

    if emb.ndim != 1:
        raise RuntimeError(
            f"Unexpected embedding output shape: {emb.shape}\n"
            f"- Model: {model_path}\n"
            "Tip: use --mode predict with the classifier model (hand_cnn.keras), or use the embedding model (hand_embedding.keras) for --mode embed."
        )

    vec = [float(x) for x in emb.tolist()]
    head_n = max(0, int(head))
    return {
        "mode": "embed",
        "model": model_path,
        "image": img_path,
        "img_size": int(img_size),
        "dim": int(len(vec)),
        "vector": vec,
        "head": vec[:head_n],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a single-image prediction or embedding extraction.")
    ap.add_argument("--model", required=True, help="Path to a .keras model")
    ap.add_argument("--image", required=True, help="Path to an input image")
    ap.add_argument("--mode", choices=["predict", "embed"], default="predict")
    ap.add_argument("--img", type=int, default=224, help="Resize side length")
    ap.add_argument("--head", type=int, default=10, help="How many embedding values to print")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    args = ap.parse_args()

    # Early path checks (better exit codes).
    if not os.path.exists(args.model):
        _eprint(f"Model not found: {args.model}")
        raise SystemExit(EXIT_MODEL_LOAD)
    if not os.path.exists(args.image):
        _eprint(f"Image not found: {args.image}")
        raise SystemExit(EXIT_BAD_IMAGE)

    try:
        if args.mode == "predict":
            out = predict_class(args.model, args.image, args.img)
        else:
            out = extract_embedding(args.model, args.image, args.img, args.head)
    except FileNotFoundError as e:
        _eprint(str(e))
        raise SystemExit(EXIT_BAD_IMAGE)
    except Exception as e:
        _eprint(str(e))
        raise SystemExit(EXIT_MODEL_LOAD)

    if args.json:
        print(json.dumps(out, ensure_ascii=False))
        raise SystemExit(EXIT_OK)

    if out.get("mode") == "predict":
        top = out.get("top") or {}
        print("Top prediction:")
        print(f"- {top.get('label')} (p={float(top.get('p', 0.0)):.4f})")
        print("\nAll class probabilities:")
        for item in out.get("probs") or []:
            print(f"- {item.get('label')}: {float(item.get('p', 0.0)):.4f}")
    else:
        print("Embedding:")
        print("- dim:", out.get("dim"))
        head_vals = out.get("head") or []
        print(f"- first {len(head_vals)} values:", np.array2string(np.array(head_vals, dtype=np.float32), precision=5, separator=", "))

    raise SystemExit(EXIT_OK)


if __name__ == "__main__":
    main()
