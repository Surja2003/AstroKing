"""Train a lightweight downstream classifier on embeddings.

Note: This file depends on scikit-learn. In this repo, ML scripts often run in a
separate Python environment (e.g. `ml/.venv`) than the backend/app. Some editors
may show unresolved-import warnings if they're pointed at a different env.
"""

# pyright: reportMissingModuleSource=false

import argparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help=".npz from extract_embeddings.py")
    ap.add_argument("--out", required=True, help="Output .pkl")
    ap.add_argument("--label_key", default="y", help="Key inside the npz to use as labels")
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    X = data["X"].astype(np.float32)

    if args.label_key not in data:
        raise SystemExit(
            f"Label key '{args.label_key}' not found in embeddings. Available keys: {list(data.keys())}. "
            "If you extracted without labels, re-run extract_embeddings.py with directory mode or CSV label_col."
        )

    y = data[args.label_key]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, random_state=args.seed, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))

    joblib.dump({"model": clf}, args.out)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
