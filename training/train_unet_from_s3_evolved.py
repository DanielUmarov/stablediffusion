#!/usr/bin/env python3
import argparse
import os
import re
import time
import math
import csv
import sys
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F

import boto3
from botocore.exceptions import ClientError

# --- repo root on path ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.unet.unet import UNet  # your UNet


# -----------------------------
# Device
# -----------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# S3 helpers
# -----------------------------
def parse_s3_uri(uri: str):
    assert uri.startswith("s3://"), f"Expected s3://..., got: {uri}"
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def s3_list_keys(bucket: str, prefix: str, suffix: str = ""):
    """List keys under prefix. No ListBuckets permission required."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if suffix and not k.endswith(suffix):
                continue
            out.append(k)
    return out


STEP_RE = re.compile(r"unet_step_(\d+)\.pt$")


def s3_find_latest_checkpoint(s3_ckpt_prefix: str):
    """
    Priority:
      1) unet_latest.pt if exists
      2) max step in unet_step_*.pt
    Returns: s3://bucket/key or None
    """
    bucket, prefix = parse_s3_uri(s3_ckpt_prefix)
    keys = s3_list_keys(bucket, prefix)
    if not keys:
        return None

    latest_key = None
    best_step = -1

    # 1) prefer unet_latest.pt
    for k in keys:
        if k.endswith("/unet_latest.pt") or Path(k).name == "unet_latest.pt":
            return f"s3://{bucket}/{k}"

    # 2) fallback: max unet_step_*.pt
    for k in keys:
        name = Path(k).name
        m = STEP_RE.match(name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                latest_key = k

    return f"s3://{bucket}/{latest_key}" if latest_key else None


def s3_download(s3_uri: str, local_path: Path):
    bucket, key = parse_s3_uri(s3_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(local_path))


def s3_upload(local_path: Path, s3_uri: str):
    bucket, key = parse_s3_uri(s3_uri)
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def s3_join(prefix_uri: str, filename: str):
    if not prefix_uri.endswith("/"):
        prefix_uri += "/"
    return prefix_uri + filename


# -----------------------------
# Diffusion schedule (simple)
# -----------------------------
def make_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


# -----------------------------
# Local shard cache (size-limited)
# -----------------------------
def get_file_size_bytes(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


class ShardCache:
    """
    Simple LRU cache of downloaded shard .pt files.
    Keeps total size <= cache_gb.
    """
    def __init__(self, cache_dir: Path, cache_gb: float):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(cache_gb * (1024**3))
        self.lru = OrderedDict()  # key -> bytes

        # initialize with existing files
        for p in sorted(self.cache_dir.glob("*.pt")):
            b = get_file_size_bytes(p)
            self.lru[p.name] = b
        self._evict_if_needed()

    def _current_bytes(self):
        return sum(self.lru.values())

    def _evict_if_needed(self):
        while self._current_bytes() > self.max_bytes and len(self.lru) > 0:
            name, _ = self.lru.popitem(last=False)  # oldest
            fp = self.cache_dir / name
            try:
                fp.unlink()
            except FileNotFoundError:
                pass

    def touch(self, name: str):
        if name in self.lru:
            b = self.lru.pop(name)
            self.lru[name] = b

    def ensure(self, s3_uri: str) -> Path:
        name = Path(parse_s3_uri(s3_uri)[1]).name  # last component
        local = self.cache_dir / name
        if local.exists():
            self.touch(name)
            return local

        # download
        s3_download(s3_uri, local)
        b = get_file_size_bytes(local)
        self.lru[name] = b
        self._evict_if_needed()
        return local


# -----------------------------
# Training
# -----------------------------
def save_checkpoint(save_path: Path, model, optimizer, step: int, scaler=None, extra: dict | None = None):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, save_path)


def append_loss_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--s3_latents_prefix", type=str, required=True,
                    help="s3://BUCKET/PREFIX/ containing shard_*.pt latents")
    ap.add_argument("--max_shards", type=int, default=0, help="0 = all shards under prefix")

    # cache
    ap.add_argument("--cache_dir", type=str, default="data/_s3_cache_latents")
    ap.add_argument("--cache_gb", type=float, default=50)

    # training
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=1000, help="diffusion timesteps")
    ap.add_argument("--print_every", type=int, default=10)

    # perf / stability
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (recommended for CUDA).")
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 disables, else clip grad norm.")
    ap.add_argument("--seed", type=int, default=0, help="0 disables; else set RNG seed.")

    # checkpoints
    ap.add_argument("--save_dir", type=str, default="checkpoints/unet")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--s3_ckpt_prefix", type=str, default="",
                    help="If set, upload checkpoints to this prefix: s3://BUCKET/PREFIX/")
    ap.add_argument("--auto_resume_s3", action="store_true",
                    help="If set and --s3_ckpt_prefix is set, auto-resume from latest checkpoint in S3.")
    ap.add_argument("--resume_local", type=str, default="",
                    help="Path to local checkpoint to resume from (overrides auto-resume).")

    # logging
    ap.add_argument("--loss_csv", type=str, default="checkpoints/unet/loss.csv")
    ap.add_argument("--csv_every", type=int, default=10, help="Write a CSV row every N steps.")

    args = ap.parse_args()

    device = pick_device()
    print("Device:", device)

    # dtype for stored latents + schedule math (keep fp32; AMP handles fp16 on CUDA)
    dtype = torch.float32  # keep float32 for MPS stability + general safety

    # CUDA speed knobs (safe defaults)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # seeding (optional)
    if args.seed and args.seed > 0:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- list shards in S3 ----
    lat_bucket, lat_prefix = parse_s3_uri(args.s3_latents_prefix)
    shard_keys = s3_list_keys(lat_bucket, lat_prefix, suffix=".pt")
    shard_keys = [k for k in shard_keys if Path(k).name.endswith(".pt")]
    shard_keys.sort()

    if args.max_shards and args.max_shards > 0:
        shard_keys = shard_keys[: args.max_shards]

    if not shard_keys:
        raise RuntimeError(f"No .pt shards found under {args.s3_latents_prefix}")

    shard_uris = [f"s3://{lat_bucket}/{k}" for k in shard_keys]
    print(f"Found {len(shard_uris)} shards")

    cache = ShardCache(Path(args.cache_dir), args.cache_gb)

    # ---- model/opt ----
    model = UNet().to(device=device, dtype=dtype).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- schedule ----
    _, _, abar = make_beta_schedule(args.T, device=device)

    # ---- resume ----
    start_step = 0
    if args.resume_local:
        ckpt_path = Path(args.resume_local)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        print("Resumed from local:", ckpt_path, "at step", start_step)

    elif args.auto_resume_s3 and args.s3_ckpt_prefix:
        latest = s3_find_latest_checkpoint(args.s3_ckpt_prefix)
        if latest:
            local_resume = Path(args.save_dir) / "_resume_from_s3.pt"
            print("Auto-resume downloading:", latest)
            s3_download(latest, local_resume)
            ckpt = torch.load(local_resume, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler"])
            start_step = int(ckpt.get("step", 0))
            print("Resumed from S3 at step", start_step)
        else:
            print("No S3 checkpoint found; starting fresh.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = Path(args.loss_csv)

    # ---- training loop ----
    shard_i = 0
    step = start_step
    t0 = time.time()

    while step < args.steps:
        # cycle shards
        shard_uri = shard_uris[shard_i % len(shard_uris)]
        shard_i += 1

        local_shard = cache.ensure(shard_uri)

        shard_ckpt = torch.load(local_shard, map_location="cpu")
        latents = shard_ckpt["latents"]  # expected: [N,4,H,W]
        if not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents)

        latents = latents.contiguous()

        # sample batch (index on CPU, then move batch)
        N = latents.shape[0]
        idx = torch.randint(0, N, (args.batch_size,))
        x0 = latents[idx].to(device=device, dtype=dtype, non_blocking=True)  # [B,4,H,W]

        # sample timesteps + noise
        t = torch.randint(0, args.T, (args.batch_size,), device=device, dtype=torch.long)
        eps = torch.randn_like(x0)

        # x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps
        abar_t = abar[t].view(-1, 1, 1, 1)  # [B,1,1,1]
        xt = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * eps

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(xt, t)
            loss = F.mse_loss(pred, eps)

        scaler.scale(loss).backward()

        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        step += 1

        # logging (print)
        if step % args.print_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            shard_name = Path(parse_s3_uri(shard_uri)[1]).name
            amp_tag = "amp" if use_amp else "fp32"
            print(f"step {step:07d} | loss {loss.item():.6f} | {amp_tag} | shard {shard_name} | {dt:.1f}s")

        # logging (csv) - throttle
        if args.csv_every > 0 and (step % args.csv_every == 0):
            append_loss_csv(loss_csv, {
                "step": step,
                "loss": float(loss.item()),
                "shard": Path(parse_s3_uri(shard_uri)[1]).name,
                "device": str(device),
                "amp": int(use_amp),
                "batch_size": int(args.batch_size),
            })

        # checkpointing
        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = save_dir / f"unet_step_{step:07d}.pt"
            latest_path = save_dir / "unet_latest.pt"

            save_checkpoint(ckpt_path, model, optimizer, step, scaler=scaler if use_amp else None, extra={"T": args.T})
            save_checkpoint(latest_path, model, optimizer, step, scaler=scaler if use_amp else None, extra={"T": args.T})
            print("saved:", ckpt_path)

            # upload to S3 if requested
            if args.s3_ckpt_prefix:
                try:
                    s3_upload(ckpt_path, s3_join(args.s3_ckpt_prefix, ckpt_path.name))
                    s3_upload(latest_path, s3_join(args.s3_ckpt_prefix, latest_path.name))
                    if loss_csv.exists():
                        s3_upload(loss_csv, s3_join(args.s3_ckpt_prefix, loss_csv.name))
                    print("uploaded to:", args.s3_ckpt_prefix)
                except ClientError as e:
                    print("WARNING: S3 upload failed:", e)


if __name__ == "__main__":
    main()
