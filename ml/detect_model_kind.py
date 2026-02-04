import argparse
import json
import sys
from typing import Any, Optional, Tuple

from tf_require import require_tensorflow


def _pick_output_shape(output_shape: Any) -> Optional[Tuple[int, ...]]:
    # output_shape can be:
    # - tuple like (None, D)
    # - list/tuple of shapes for multi-output
    # - dict-like not represented here (Keras usually returns list)
    if output_shape is None:
        return None

    if isinstance(output_shape, tuple):
        return output_shape

    if isinstance(output_shape, list) and output_shape:
        # Prefer a 2D (None, D) output if present.
        for s in output_shape:
            if isinstance(s, tuple) and len(s) == 2 and s[1] is not None:
                return s
        first = output_shape[0]
        return first if isinstance(first, tuple) else None

    return None


def detect(model_path: str) -> dict:
    tf = require_tensorflow()
    model = tf.keras.models.load_model(model_path)

    output_shape = getattr(model, "output_shape", None)
    chosen = _pick_output_shape(output_shape)

    out_dim = None
    if chosen and len(chosen) >= 2:
        out_dim = chosen[-1]

    # Heuristic:
    # - Classifier head: small number of classes (usually 2)
    # - Embedding model: typically 64/128/256/etc.
    mode = "predict"
    if isinstance(out_dim, int) and out_dim > 10:
        mode = "embed"

    return {
        "mode": mode,
        "output_shape": output_shape,
        "out_dim": out_dim,
        "model": model_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect whether a .keras model is a classifier or embedding model.")
    ap.add_argument("--model", required=True, help="Path to a .keras model")
    args = ap.parse_args()

    try:
        info = detect(args.model)
        print(json.dumps(info, ensure_ascii=False))
    except FileNotFoundError:
        print(f"Model not found: {args.model}", file=sys.stderr)
        raise SystemExit(3)
    except Exception as e:
        print(f"Failed to inspect model: {args.model}\n{e}", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
