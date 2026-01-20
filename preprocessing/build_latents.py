#!/usr/bin/env python3
"""
build_latents.py

Encodes images into Stable Diffusion VAE latents (fp16 by default) and saves them in shards.

Input:
- data/processed/pairs.tsv  (image_path<TAB>caption)
  OR
- data/processed/filelist.txt (one image path per line)

Output (example):
data/latents/sd15-vae/cc12m_256/
  shard_00000.pt
  shard_00001.pt
  ...
  manifest.jsonl   (maps each row -> shard + index + caption + original path)

Notes:
- Uses center-crop + resize to 256.
- Stores latents as fp16 by default to save space.
- VAE scaling factor: diffusers uses latents = posterior.sample() * 0.18215 for SD 1.x.
- Supports --resume by reading manifest.jsonl and skipping already-encoded paths.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from diffusers import AutoencoderKL

ImageFile.LOAD_TRUNCATED_IMAGES = False  # be strict


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pairs(pairs_path: Path):
    rows = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            img, cap = line.split("\t", 1)
            rows.append((img, cap))
    return rows


def load_filelist(filelist_path: Path):
    rows = []
    with filelist_path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                rows.append((p, ""))  # empty caption
    return rows


def center_crop_resize(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((size, size), resample=Image.BICUBIC)
    return im


def pil_to_tensor(im: Image.Image) -> torch.Tensor:
    # [H,W,C] uint8 -> float32 in [0,1]
    x = torch.from_numpy(np.array(im)).float() / 255.0
    # [H,W,C] -> [C,H,W]
    x = x.permute(2, 0, 1)
    # [0,1] -> [-1,1]
    x = x * 2.0 - 1.0
    return x


def next_shard_index(out_dir: Path) -> int:
    """
    If shards already exist, continue numbering from the next available index.
    shard_00000.pt -> returns 1, etc.
    """
    pat = re.compile(r"shard_(\d{5})\.pt$")
    max_idx = -1
    for p in out_dir.glob("shard_*.pt"):
        m = pat.search(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default="data/processed/pairs.tsv", help="image<TAB>caption")
    ap.add_argument("--filelist", type=str, default="", help="fallback: one image path per line")
    ap.add_argument("--out_dir", type=str, default="data/latents/sd15-vae/cc12m_256", help="where to write shards")
    ap.add_argument("--vae_id", type=str, default="runwayml/stable-diffusion-v1-5", help="HF model id (VAE is read from it)")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shard_size", type=int, default=2000, help="how many samples per shard file")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--max_items", type=int, default=0, help="0 = all")
    ap.add_argument("--resume", action="store_true", help="skip items already present in manifest.jsonl")

    args = ap.parse_args()

    device = pick_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Load input rows
    pairs_path = Path(args.pairs)
    if pairs_path.exists():
        rows = load_pairs(pairs_path)
        source_kind = "pairs"
    else:
        if not args.filelist:
            raise SystemExit("pairs.tsv not found and --filelist not provided.")
        rows = load_filelist(Path(args.filelist))
        source_kind = "filelist"

    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]

    # Resume: skip items already written to manifest.jsonl
    done_paths = set()
    if args.resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as mf:
            for line in mf:
                try:
                    rec = json.loads(line)
                    done_paths.add(rec["path"])
                except Exception:
                    continue
        print(f"[resume] found {len(done_paths)} already-encoded samples in {manifest_path}")

        rows = [(p, c) for (p, c) in rows if p not in done_paths]
        print(f"[resume] remaining items={len(rows)}")

    print(f"[input] source={source_kind} items={len(rows)}")
    print(f"[device] {device}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.vae_id, subfolder="vae")
    vae.eval().to(device)

    use_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    # On MPS, fp16 is supported; if you hit instability, switch to fp32.
    vae.to(dtype=use_dtype)

    # SD1.x latent scaling factor
    latent_scale = 0.18215

    # Continue shard numbering if shards already exist
    shard_idx = next_shard_index(out_dir)

    # Sharding state
    shard_items = []
    shard_caps = []
    shard_paths = []
    total_written = 0

    def flush_shard():
        nonlocal shard_idx, shard_items, shard_caps, shard_paths, total_written
        if not shard_items:
            return

        latents = torch.stack(shard_items, dim=0).cpu()  # [N,4,32,32] for 256
        caps = shard_caps
        paths = shard_paths

        shard_file = out_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(
            {
                "latents": latents,     # fp16 or fp32
                "captions": caps,       # list[str]
                "paths": paths,         # list[str]
                "image_size": args.image_size,
                "vae_id": args.vae_id,
                "latent_scale": latent_scale,
            },
            shard_file,
        )

        # manifest rows: each sample gets shard + index
        with manifest_path.open("a", encoding="utf-8") as mf:
            for i, (p, c) in enumerate(zip(paths, caps)):
                mf.write(
                    json.dumps(
                        {"shard": shard_file.name, "index": i, "path": p, "caption": c},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        total_written += len(shard_items)
        print(f"[write] {shard_file}  (+{len(shard_items)} samples, total={total_written})")

        shard_idx += 1
        shard_items, shard_caps, shard_paths = [], [], []

    # Batch encode loop
    bs = args.batch_size
    for start in tqdm(range(0, len(rows), bs), desc="encoding"):
        batch = rows[start : start + bs]
        imgs = []
        caps = []
        paths = []

        for img_path, cap in batch:
            p = Path(img_path)
            try:
                with Image.open(p) as im:
                    im = center_crop_resize(im, args.image_size)
                    x = pil_to_tensor(im)
            except Exception:
                # Skip failures to keep pipeline moving (scan_and_filter should catch most issues)
                continue

            imgs.append(x)
            caps.append(cap)
            paths.append(str(p))

        if not imgs:
            continue

        x = torch.stack(imgs, dim=0).to(device=device, dtype=use_dtype)  # [B,3,256,256]

        # Encode: posterior -> sample -> scale
        posterior = vae.encode(x).latent_dist
        z = posterior.sample() * latent_scale  # [B,4,32,32]

        # Store per-sample (easier to shard cleanly)
        for i in range(z.shape[0]):
            shard_items.append(z[i].detach())
            shard_caps.append(caps[i])
            shard_paths.append(paths[i])

            if len(shard_items) >= args.shard_size:
                flush_shard()

    flush_shard()
    print("[done] latent encoding complete")


if __name__ == "__main__":
    main()
