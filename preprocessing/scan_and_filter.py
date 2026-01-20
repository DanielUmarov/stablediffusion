#!/usr/bin/env python3
import argparse
import hashlib
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False  # be strict


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def sha1_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_image_ext(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def verify_image(path: Path):
    """
    Returns (ok: bool, reason: str, width: int|None, height: int|None)
    """
    try:
        with Image.open(path) as im:
            im.verify()  # quick integrity check (doesn't decode fully)
        # reopen to read size (verify() leaves file in unusable state)
        with Image.open(path) as im2:
            w, h = im2.size
        return True, "ok", w, h
    except Exception as e:
        return False, f"corrupt_or_unreadable:{type(e).__name__}", None, None


def decode_test(path: Path):
    """
    Fully decodes one image to ensure it can be loaded into memory.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.load()
        return im.size


def within_aspect(w: int, h: int, min_ar: float, max_ar: float) -> bool:
    ar = w / h if h != 0 else 0.0
    return (ar >= min_ar) and (ar <= max_ar)


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            return
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser(description="Scan data/raw, validate images, filter, and build filelist.")
    ap.add_argument("--raw_dir", type=str, default="data/raw", help="Input directory to scan recursively")
    ap.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")
    ap.add_argument("--target_size", type=int, default=256, help="Target size (informational for now; used for min-size filtering)")
    ap.add_argument("--min_side", type=int, default=256, help="Reject images with min(width,height) < min_side")
    ap.add_argument("--min_ar", type=float, default=0.5, help="Min aspect ratio w/h")
    ap.add_argument("--max_ar", type=float, default=2.0, help="Max aspect ratio w/h")
    ap.add_argument("--dedupe", action="store_true", help="Remove exact duplicates by SHA1")
    ap.add_argument("--store_mode", choices=["copy", "symlink", "none"], default="none",
                    help="Store valid images into out_dir/images by copy/symlink, or none (keep in place)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--random_load_n", type=int, default=100, help="Random load test count at end")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_images_dir = out_dir / "images"
    filelist_path = out_dir / "filelist.txt"

    if not raw_dir.exists():
        raise SystemExit(f"raw_dir does not exist: {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidate files
    candidates = [p for p in raw_dir.rglob("*") if p.is_file() and is_image_ext(p)]
    print(f"[scan] found {len(candidates)} candidate files in {raw_dir}")

    reasons = Counter()
    valid = []
    seen_hashes = set()

    for p in tqdm(candidates, desc="validating"):
        ok, reason, w, h = verify_image(p)
        if not ok:
            reasons[reason] += 1
            continue

        # filters
        if min(w, h) < args.min_side:
            reasons["too_small"] += 1
            continue

        if not within_aspect(w, h, args.min_ar, args.max_ar):
            reasons["bad_aspect_ratio"] += 1
            continue

        if args.dedupe:
            try:
                hsh = sha1_of_file(p)
            except Exception as e:
                reasons[f"hash_fail:{type(e).__name__}"] += 1
                continue
            if hsh in seen_hashes:
                reasons["duplicate_sha1"] += 1
                continue
            seen_hashes.add(hsh)

        valid.append(p)

    # Write outputs (store + filelist)
    stored_paths = []
    if args.store_mode != "none":
        for src in tqdm(valid, desc=f"storing ({args.store_mode})"):
            # Keep relative layout under raw_dir
            rel = src.relative_to(raw_dir)
            dst = out_images_dir / rel
            copy_or_link(src, dst, args.store_mode)
            stored_paths.append(dst)
        list_paths = stored_paths
    else:
        list_paths = valid

    # filelist.txt: one path per line
    with filelist_path.open("w", encoding="utf-8") as f:
        for p in list_paths:
            f.write(str(p) + "\n")

    # Summary
    rejected = sum(reasons.values())
    print("\n===== SUMMARY =====")
    print(f"raw_dir: {raw_dir}")
    print(f"out_dir: {out_dir}")
    print(f"target_size (planned): {args.target_size}")
    print(f"candidates: {len(candidates)}")
    print(f"valid: {len(valid)}")
    print(f"rejected: {rejected}")
    if reasons:
        print("\nRejection reasons:")
        for k, v in reasons.most_common():
            print(f"  {k}: {v}")
    print(f"\nWrote: {filelist_path} ({len(list_paths)} lines)")

    # Random load test
    if len(list_paths) == 0:
        raise SystemExit("No valid images found. Check your raw data or relax filters.")

    random.seed(args.seed)
    n = min(args.random_load_n, len(list_paths))
    sample = random.sample(list_paths, n)

    print(f"\n[random_load_test] loading {n} random images...")
    for p in tqdm(sample, desc="loading"):
        try:
            decode_test(p)
        except Exception as e:
            raise SystemExit(f"Random load test failed on: {p}\nError: {type(e).__name__}: {e}")

    print("[random_load_test] success ✅")


if __name__ == "__main__":
    main()
