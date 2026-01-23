#!/usr/bin/env python3
# training/train_unet_from_s3.py

import sys
from pathlib import Path

# --- FORCE repo root onto python path (so "models" imports work) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import hashlib
import os
import random
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import boto3

from models.unet.unet import UNet


# -------------------------
# device
# -------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# S3 helpers
# -------------------------
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("s3_prefix must start with s3://")
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def list_pt_keys(bucket: str, prefix: str, max_keys: int | None = None) -> List[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".pt"):
                keys.append(k)
                if max_keys is not None and len(keys) >= max_keys:
                    return keys
    return keys


def key_to_cache_path(cache_dir: Path, key: str) -> Path:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.pt"


def ensure_download(bucket: str, key: str, cache_path: Path) -> Path:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        os.utime(cache_path, None)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".pt.tmp")

    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(tmp))
    tmp.replace(cache_path)
    os.utime(cache_path, None)
    return cache_path


def enforce_cache_limit(cache_dir: Path, max_gb: float):
    if max_gb <= 0:
        return
    max_bytes = int(max_gb * (1024**3))

    files = sorted(
        [p for p in cache_dir.glob("*.pt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,  # oldest first
    )
    total = sum(p.stat().st_size for p in files)
    if total <= max_bytes:
        return

    for p in files:
        try:
            sz = p.stat().st_size
            p.unlink(missing_ok=True)
            total -= sz
            if total <= max_bytes:
                break
        except Exception:
            pass


# -------------------------
# DDPM schedule
# -------------------------
def make_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device="cpu",
    dtype=torch.float32,
):
    betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def q_sample(z0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor):
    eps = torch.randn_like(z0)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    zt = torch.sqrt(ab) * z0 + torch.sqrt(1.0 - ab) * eps
    return zt, eps


def load_latents(local_pt: Path):
    # NOTE: shards are yours; this warning is about pickle safety.
    ckpt = torch.load(local_pt, map_location="cpu")
    latents = ckpt["latents"]
    meta = {
        "latent_scale": ckpt.get("latent_scale", 0.18215),
        "vae_id": ckpt.get("vae_id", None),
        "image_size": ckpt.get("image_size", None),
    }
    return latents, meta


def save_checkpoint(save_dir: Path, step: int, model: UNet, opt: AdamW, extra: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"unet_step_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            **extra,
        },
        path,
    )
    latest = save_dir / "unet_latest.pt"
    try:
        if latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)
    except Exception:
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                **extra,
            },
            latest,
        )
    print("saved:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3_prefix", type=str, required=True, help="e.g. s3://bucket/path/to/latents/")
    ap.add_argument("--cache_dir", type=str, default="data/_s3_cache_latents")
    ap.add_argument("--cache_gb", type=float, default=30.0, help="max cache size in GB (old files evicted)")
    ap.add_argument("--max_shards", type=int, default=0, help="0 = no limit; else limit number of shards listed")
    ap.add_argument("--save_dir", type=str, default="checkpoints/unet")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default="", help="path to checkpoint to resume from (optional)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)

    bucket, prefix = parse_s3_uri(args.s3_prefix)

    max_keys = None if args.max_shards == 0 else args.max_shards
    keys = list_pt_keys(bucket, prefix, max_keys=max_keys)
    if not keys:
        raise RuntimeError(f"No .pt shards found under {args.s3_prefix}")

    print("Using repo root:", REPO_ROOT)
    print("sys.path[0]:", sys.path[0])
    print("device:", device)
    print("bucket:", bucket)
    print("prefix:", prefix)
    print("num_shards_listed:", len(keys))
    print("cache_dir:", cache_dir)
    print("cache_gb:", float(args.cache_gb))

    # Download one shard to infer shape
    k0 = keys[0]
    p0 = ensure_download(bucket, k0, key_to_cache_path(cache_dir, k0))
    lat0, meta0 = load_latents(p0)
    in_channels = lat0.shape[1]

    print("example latent:", tuple(lat0.shape), "dtype:", lat0.dtype)
    print("example meta:", meta0)

    # -------------------------
    # IMPORTANT: MPS stability -> run compute in float32
    # -------------------------
    model = UNet(in_channels=in_channels).to(device).to(dtype=torch.float32)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # schedule in float32 on device
    _, _, alpha_bar = make_beta_schedule(
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
        dtype=torch.float32,
    )

    start_step = 1
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0)) + 1
        print("resumed from:", args.resume, "-> start_step:", start_step)

    current_key = None
    current_latents = None

    model.train()
    t0 = time.time()

    for step in range(start_step, args.steps + 1):
        # choose shard
        key = random.choice(keys)
        if key != current_key:
            local = ensure_download(bucket, key, key_to_cache_path(cache_dir, key))
            enforce_cache_limit(cache_dir, args.cache_gb)
            current_latents, _ = load_latents(local)
            current_key = key

        # sample batch (CAST TO FLOAT32 for MPS)
        N = current_latents.shape[0]
        idx = torch.randint(0, N, (args.batch_size,), dtype=torch.long)
        z0 = current_latents[idx].to(device, dtype=torch.float32)

        # sample timesteps
        t = torch.randint(0, args.T, (args.batch_size,), device=device, dtype=torch.long)

        # forward diffusion + target noise
        zt, eps = q_sample(z0, t, alpha_bar)

        eps_hat = model(zt, t)
        loss = F.mse_loss(eps_hat, eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.print_every == 0:
            dt = time.time() - t0
            print(f"step {step:07d} | loss {loss.item():.6f} | shard {Path(current_key).name} | {dt:.1f}s")
            t0 = time.time()

        if step % args.save_every == 0:
            save_checkpoint(
                save_dir,
                step,
                model,
                opt,
                extra={
                    "in_channels": in_channels,
                    "T": args.T,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end,
                    "example_meta": meta0,
                    "s3_prefix": args.s3_prefix,
                    "train_dtype": "float32",
                },
            )

    save_checkpoint(
        save_dir,
        args.steps,
        model,
        opt,
        extra={
            "in_channels": in_channels,
            "T": args.T,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "example_meta": meta0,
            "s3_prefix": args.s3_prefix,
            "train_dtype": "float32",
        },
    )


if __name__ == "__main__":
    main()
