import argparse
import os
from typing import List, Tuple

from eval_model import evaluate_model


def _short_name(p: str) -> str:
    p = os.path.normpath(p)
    base = os.path.basename(p)
    # If it's always hand_cnn.keras, show parent folder name too.
    if base.lower() in {"hand_cnn.keras", "model.keras", "model.h5", "hand_cnn.h5"}:
        parent = os.path.basename(os.path.dirname(p))
        if parent:
            return f"{parent}/{base}"
    return base


def _print_table(rows: List[Tuple[str, float, float, float, float]]) -> None:
    headers = ("Model", "Acc", "Prec(m)", "Rec(m)", "F1(m)")
    col0 = max(len(headers[0]), max((len(r[0]) for r in rows), default=5))

    def fmt(v: float) -> str:
        return f"{v:.4f}"

    line = f"{headers[0]:<{col0}}  {headers[1]:>7}  {headers[2]:>7}  {headers[3]:>7}  {headers[4]:>7}"
    print(line)
    print("-" * len(line))
    for name, acc, pm, rm, f1m in rows:
        print(f"{name:<{col0}}  {fmt(acc):>7}  {fmt(pm):>7}  {fmt(rm):>7}  {fmt(f1m):>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare multiple saved models on the same dataset (no retraining).")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model paths (.keras/.h5). Example: --models m1.keras m2.keras",
    )
    ap.add_argument(
        "--data",
        required=True,
        help="Dataset directory (root with classes OR split root with train/val/test).",
    )
    ap.add_argument("--subset", choices=["auto", "root", "train", "val", "test"], default="auto")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--show_cm", action="store_true", help="Also print a confusion matrix per model (if sklearn is available).")
    args = ap.parse_args()

    rows: List[Tuple[str, float, float, float, float]] = []

    for model_path in args.models:
        metrics, cm, class_names = evaluate_model(
            model_path=model_path,
            data_root=args.data,
            subset=args.subset,
            img=args.img,
            batch=args.batch,
            seed=args.seed,
            quiet=True,
        )

        rows.append(
            (
                _short_name(model_path),
                float(metrics["accuracy"]),
                float(metrics["precision_macro"]),
                float(metrics["recall_macro"]),
                float(metrics["f1_macro"]),
            )
        )

        if args.show_cm and cm is not None:
            print(f"\nModel: {model_path}")
            print(f"Classes: {class_names}")
            print("Confusion matrix (rows=true, cols=pred):")
            print(cm)

    # Sort best-first by macro F1, tie-breaker accuracy.
    rows.sort(key=lambda r: (r[4], r[1]), reverse=True)

    print("\nResults")
    _print_table(rows)


if __name__ == "__main__":
    main()
