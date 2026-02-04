import argparse
import os
import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_files(folder: str) -> Iterable[str]:
    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                yield os.path.join(root, name)


def _sha1(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _detect_layout(root: str) -> str:
    # Returns "split" (train/val/test) or "flat" (class folders only)
    if os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "val")):
        return "split"
    return "flat"


def _class_folders(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset sanity report (counts, balance, duplicates across splits).")
    ap.add_argument("--root", required=True, help="Dataset root")
    ap.add_argument("--classes", default="", help="Comma-separated expected classes (optional)")
    ap.add_argument("--hash", action="store_true", help="Compute file hashes to detect duplicates across splits")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    layout = _detect_layout(root)

    expected = [c.strip() for c in args.classes.split(",") if c.strip()]

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    paths_by_split: Dict[str, List[str]] = defaultdict(list)

    if layout == "split":
        splits = ["train", "val", "test"]
        for split in splits:
            split_root = os.path.join(root, split)
            if not os.path.isdir(split_root):
                continue
            classes = _class_folders(split_root)
            if expected and set(classes) != set(expected):
                print(f"Warning: classes under {split} are {classes}, expected {expected}")
            for cls in classes:
                cls_root = os.path.join(split_root, cls)
                n = sum(1 for _ in _iter_files(cls_root))
                counts[(split, cls)] += n
                paths_by_split[split].extend(list(_iter_files(cls_root)))
    else:
        split = "all"
        classes = _class_folders(root)
        if expected and set(classes) != set(expected):
            print(f"Warning: classes are {classes}, expected {expected}")
        for cls in classes:
            cls_root = os.path.join(root, cls)
            n = sum(1 for _ in _iter_files(cls_root))
            counts[(split, cls)] += n
            paths_by_split[split].extend(list(_iter_files(cls_root)))

    print(f"Dataset root: {root}")
    print(f"Layout: {layout}")

    # Pretty print counts
    splits_seen = sorted({k[0] for k in counts.keys()})
    classes_seen = sorted({k[1] for k in counts.keys()})

    for split in splits_seen:
        total = sum(counts[(split, cls)] for cls in classes_seen if (split, cls) in counts)
        print(f"\n{split}: total={total}")
        for cls in classes_seen:
            if (split, cls) in counts:
                print(f"- {cls}: {counts[(split, cls)]}")

    if args.hash and layout == "split":
        # Detect duplicates across splits by content hash.
        print("\nDuplicate check (by SHA1):")
        hashes_by_split: Dict[str, Dict[str, str]] = {}
        for split, files in paths_by_split.items():
            m: Dict[str, str] = {}
            for p in files:
                try:
                    m[_sha1(p)] = p
                except Exception:
                    continue
            hashes_by_split[split] = m

        splits = [s for s in ("train", "val", "test") if s in hashes_by_split]
        dup_pairs = 0
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                a, b = splits[i], splits[j]
                common = set(hashes_by_split[a].keys()).intersection(hashes_by_split[b].keys())
                if common:
                    dup_pairs += len(common)
                    print(f"- {a} vs {b}: {len(common)} duplicates")
        if dup_pairs == 0:
            print("- none found")


if __name__ == "__main__":
    main()
