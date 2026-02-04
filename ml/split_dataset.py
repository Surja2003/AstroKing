import argparse
import os
import random
import shutil
from typing import Iterable, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(folder: str) -> Iterable[str]:
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            yield name


def _split_counts(n: int, train: float, val: float, test: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    # Normalize in case the user passes 70/15/15 etc.
    total = train + val + test
    if total <= 0:
        raise ValueError("train/val/test fractions must sum to > 0")
    train, val, test = train / total, val / total, test / total

    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def main() -> None:
    ap = argparse.ArgumentParser(description="Split a folder-per-class image dataset into train/val/test folders.")
    ap.add_argument("--source", default=None, help="Dataset root containing class folders (e.g. male/, female/)")
    ap.add_argument("--target", default=None, help="Output folder (will create train/val/test subfolders)")
    # Friendly aliases (used by other scripts/docs)
    ap.add_argument("--data_dir", dest="source", help="Alias for --source")
    ap.add_argument("--out_dir", dest="target", help="Alias for --target")
    ap.add_argument("--classes", default="male,female", help="Comma-separated class folder names")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--copy", action="store_true", help="Copy files (default)")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy")
    ap.add_argument("--hardlink", action="store_true", help="Create hardlinks (saves disk; same drive only)")
    args = ap.parse_args()

    if not args.source or not args.target:
        raise SystemExit("Missing required args: --source/--target (or --data_dir/--out_dir)")

    source = os.path.abspath(args.source)
    target = os.path.abspath(args.target)

    mode_flags = sum(1 for b in (args.move, args.hardlink) if b)
    if mode_flags > 1:
        raise SystemExit("Choose only one of --move or --hardlink")

    do_move = bool(args.move)
    do_hardlink = bool(args.hardlink)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        raise SystemExit("--classes must contain at least one class")

    random.seed(args.seed)

    for split in ("train", "val", "test"):
        for cls in classes:
            os.makedirs(os.path.join(target, split, cls), exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(source, cls)
        if not os.path.isdir(cls_path):
            raise SystemExit(f"Missing class folder: {cls_path}")

        files = list(_iter_images(cls_path))
        if not files:
            raise SystemExit(f"No images found under: {cls_path}")

        random.shuffle(files)
        n_train, n_val, n_test = _split_counts(len(files), args.train, args.val, args.test)

        split_files = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split, names in split_files.items():
            for name in names:
                src = os.path.join(cls_path, name)
                dst = os.path.join(target, split, cls, name)
                if do_move:
                    shutil.move(src, dst)
                elif do_hardlink:
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)

        print(f"{cls}: total={len(files)} train={n_train} val={n_val} test={n_test}")

    print("Done.")
    print("Target:", target)


if __name__ == "__main__":
    main()
