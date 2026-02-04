"""Extract a subset of the HandInfo dataset into a folder-per-class layout.

This dataset (as provided in the user's archive) ships with a HandInfo.csv file
containing metadata like `gender` and `aspectOfHand` (e.g. "palmar left",
"dorsal right").

Typical use cases:
- Build a PALMAR-only gender dataset for training (reduces view variation).
- Build a PALMAR-vs-DORSAL dataset for view classification.

Output layout examples:

1) Gender classification (label_source=gender)
   out_dir/
     FEMALE/*.jpg
     MALE/*.jpg

2) View classification (label_source=aspect)
   out_dir/
     PALMAR/*.jpg
     DORSAL/*.jpg

By default, this script copies files. You can also hardlink to save space.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Filters:
    require_view: Optional[str]  # 'palmar'|'dorsal'|None
    require_side: Optional[str]  # 'left'|'right'|None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _parse_aspect(aspect_of_hand: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (view, side) from aspectOfHand."""
    s = _norm(aspect_of_hand)
    view = None
    side = None

    if "palmar" in s or "palm" in s:
        view = "palmar"
    elif "dorsal" in s or "back" in s:
        view = "dorsal"

    if "left" in s:
        side = "left"
    elif "right" in s:
        side = "right"

    return view, side


def _match_filters(aspect_of_hand: str, filters: Filters) -> bool:
    view, side = _parse_aspect(aspect_of_hand)

    if filters.require_view is not None and view != filters.require_view:
        return False
    if filters.require_side is not None and side != filters.require_side:
        return False

    return True


def _label_from_gender(gender: str) -> Optional[str]:
    g = _norm(gender)
    if g in {"male", "m"}:
        return "MALE"
    if g in {"female", "f"}:
        return "FEMALE"
    return None


def _label_from_aspect(aspect_of_hand: str) -> Optional[str]:
    view, _ = _parse_aspect(aspect_of_hand)
    if view == "palmar":
        return "PALMAR"
    if view == "dorsal":
        return "DORSAL"
    return None


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hardlink_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            # Windows supports hardlinks for files on the same volume.
            os.link(src, dst)
            return
        except OSError:
            # Fall back to copy.
            shutil.copy2(src, dst)
            return

    raise ValueError(f"Unknown mode: {mode}")


def extract(
    *,
    handinfo_csv: Path,
    images_dir: Path,
    out_dir: Path,
    label_source: str,
    filters: Filters,
    mode: str,
    limit: Optional[int],
) -> Dict[str, int]:
    if not handinfo_csv.exists():
        raise FileNotFoundError(handinfo_csv)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    _safe_mkdir(out_dir)

    label_counts: Dict[str, int] = {}
    copied = 0
    missing = 0
    skipped = 0

    with handinfo_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"imageName", "aspectOfHand"}
        if label_source == "gender":
            required.add("gender")

        missing_cols = sorted(required - set(reader.fieldnames or []))
        if missing_cols:
            raise ValueError(
                f"HandInfo CSV missing columns: {missing_cols}. Present: {reader.fieldnames}"
            )

        for row in reader:
            if limit is not None and copied >= limit:
                break

            img_name = (row.get("imageName") or "").strip()
            aspect = (row.get("aspectOfHand") or "").strip()
            if not img_name:
                skipped += 1
                continue

            if not _match_filters(aspect, filters):
                skipped += 1
                continue

            if label_source == "gender":
                label = _label_from_gender(row.get("gender") or "")
            elif label_source == "aspect":
                label = _label_from_aspect(aspect)
            else:
                raise ValueError("label_source must be 'gender' or 'aspect'")

            if not label:
                skipped += 1
                continue

            src = images_dir / img_name
            if not src.exists():
                # Sometimes datasets ship images with different casing.
                alt = next((p for p in images_dir.glob(img_name.lower())), None)
                if alt is not None:
                    src = alt
                else:
                    missing += 1
                    continue

            if src.suffix.lower() not in IMAGE_EXTS:
                skipped += 1
                continue

            dst_dir = out_dir / label
            _safe_mkdir(dst_dir)
            dst = dst_dir / src.name
            _hardlink_or_copy(src, dst, mode=mode)

            label_counts[label] = label_counts.get(label, 0) + 1
            copied += 1

    print("Done.")
    print(f"Copied/linked: {copied}")
    print(f"Missing files: {missing}")
    print(f"Skipped rows:  {skipped}")
    print("Label counts:")
    for k in sorted(label_counts):
        print(f"  {k}: {label_counts[k]}")

    return label_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a labeled subset from HandInfo.csv into folder-per-class directories."
    )
    parser.add_argument(
        "--handinfo_csv",
        required=True,
        help="Path to HandInfo.csv (e.g. C:/Users/.../archive/HandInfo.csv)",
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Path to the images directory (e.g. .../archive/Hands/Hands)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output folder (will contain class subfolders)",
    )
    parser.add_argument(
        "--label_source",
        choices=["gender", "aspect"],
        default="gender",
        help="Which column/label to use for class folders.",
    )

    view = parser.add_mutually_exclusive_group()
    view.add_argument(
        "--palmar",
        action="store_true",
        help="Keep only palmar (front/palm) images.",
    )
    view.add_argument(
        "--dorsal",
        action="store_true",
        help="Keep only dorsal (back of hand) images.",
    )

    side = parser.add_mutually_exclusive_group()
    side.add_argument("--left", action="store_true", help="Keep only left hands.")
    side.add_argument("--right", action="store_true", help="Keep only right hands.")

    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to materialize the output dataset (hardlink saves disk).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for debugging (e.g. 200).",
    )

    args = parser.parse_args()

    require_view = None
    if args.palmar:
        require_view = "palmar"
    elif args.dorsal:
        require_view = "dorsal"

    require_side = None
    if args.left:
        require_side = "left"
    elif args.right:
        require_side = "right"

    extract(
        handinfo_csv=Path(args.handinfo_csv),
        images_dir=Path(args.images_dir),
        out_dir=Path(args.out_dir),
        label_source=args.label_source,
        filters=Filters(require_view=require_view, require_side=require_side),
        mode=args.mode,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
