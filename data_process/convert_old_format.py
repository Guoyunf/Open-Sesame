#!/usr/bin/env python3
"""Convert legacy flat recording layout to the new episode-based format."""

import argparse
import os
import re
import shutil
from pathlib import Path


IMG_RE = re.compile(r"v(?P<cam>[12])_(?P<idx>\d+)_(?P<tag>.+)\.png")


def convert(root: str) -> None:
    root_path = Path(root)
    pos_files = sorted(root_path.glob("pos_*.npy"))
    for pos in pos_files:
        tag = pos.stem.split("pos_")[1]
        tag_dir = root_path / tag
        for sub in ["cam1/color", "cam1/depth", "cam2/color", "cam2/depth"]:
            (tag_dir / sub).mkdir(parents=True, exist_ok=True)
        shutil.move(str(pos), tag_dir / "pos.npy")

    for img in root_path.glob("color/v*_*.png"):
        m = IMG_RE.match(img.name)
        if not m:
            continue
        cam = m.group("cam")
        idx = int(m.group("idx"))
        tag = m.group("tag")
        dest = root_path / tag / f"cam{cam}" / "color" / f"{idx:06d}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), dest)
        depth_src = root_path / "depth" / img.name
        if depth_src.exists():
            destd = root_path / tag / f"cam{cam}" / "depth" / f"{idx:06d}.png"
            destd.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(depth_src), destd)

    # cleanup
    shutil.rmtree(root_path / "color", ignore_errors=True)
    shutil.rmtree(root_path / "depth", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy recording format")
    parser.add_argument("root", nargs="?", default="data_record", help="root directory of recordings")
    args = parser.parse_args()
    convert(args.root)


if __name__ == "__main__":
    main()

