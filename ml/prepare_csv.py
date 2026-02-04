import argparse
import csv
import os
import shutil
from typing import Dict, Optional


def _sniff_dialect(path: str) -> csv.Dialect:
    # Supports common CSV/TSV formats without requiring pandas.
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
    except Exception:
        return csv.get_dialect("excel")


def _safe_label(value: str) -> str:
    # Normalize label to a folder name.
    v = str(value).strip().lower()
    v = v.replace("/", "_").replace("\\", "_")
    v = "_".join(v.split())
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a labels CSV into folder-per-class layout.")
    ap.add_argument("--csv", required=True, help="Path to labels CSV/TSV")
    ap.add_argument("--images", required=True, help="Folder with images (or a root containing nested paths)")
    ap.add_argument("--out", required=True, help="Output dataset root (creates <out>/<label>/ files)")
    ap.add_argument("--label_col", default="gender", help="Label column name")
    ap.add_argument("--file_col", default="filename", help="Filename/path column name")
    ap.add_argument("--copy", action="store_true", help="Copy files (default)")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy")
    ap.add_argument("--skip_missing", action="store_true", help="Skip missing images instead of failing")
    args = ap.parse_args()

    if args.move and args.copy:
        raise SystemExit("Choose only one of --copy or --move")

    do_move = bool(args.move)

    csv_path = os.path.abspath(args.csv)
    images_root = os.path.abspath(args.images)
    out_root = os.path.abspath(args.out)

    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    if not os.path.isdir(images_root):
        raise SystemExit(f"Images folder not found: {images_root}")

    os.makedirs(out_root, exist_ok=True)

    dialect = _sniff_dialect(csv_path)

    counts: Dict[str, int] = {}
    missing = 0
    total = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise SystemExit("CSV has no header row")

        for row in reader:
            total += 1
            if args.label_col not in row or args.file_col not in row:
                raise SystemExit(
                    f"CSV must contain columns '{args.file_col}' and '{args.label_col}'. "
                    f"Found: {reader.fieldnames}"
                )

            label = _safe_label(row[args.label_col])
            rel_path = str(row[args.file_col]).strip()
            if not rel_path:
                continue

            src = os.path.join(images_root, rel_path)
            if not os.path.exists(src):
                missing += 1
                if args.skip_missing:
                    continue
                raise SystemExit(f"Missing image: {src}")

            dst_dir = os.path.join(out_root, label)
            os.makedirs(dst_dir, exist_ok=True)

            dst = os.path.join(dst_dir, os.path.basename(rel_path))
            if do_move:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)

            counts[label] = counts.get(label, 0) + 1

    print("Prepared dataset:")
    for label in sorted(counts.keys()):
        print(f"- {label}: {counts[label]}")

    if missing:
        print(f"Missing images skipped: {missing}")

    print("Output:", out_root)


if __name__ == "__main__":
    main()
