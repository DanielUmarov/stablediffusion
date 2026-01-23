#!/usr/bin/env python3
import argparse
import os
import re
import time
import csv
import sys
from pathlib import Path
from collections import OrderedDict
import random

import torch
import torch.nn.functional as F

import boto3
from botocore.exceptions import ClientError

# --- repo root on path ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.unet.unet import UNet


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


def s3_list_common_prefixes(bucket: str, prefix: str):
    """
    List immediate subfolders under a prefix (e.g. cc12m_256_00000/)
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    prefixes = []
    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/",
    ):
        for p in page.get("CommonPrefixes", []):
            prefixes.append(p["Prefix"])
    return prefixes


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
# Diffusion schedule
# -----------------------------
def make_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


# -----------------------------
# Shard cache
# -----------------------------
def get_file_size_bytes(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


class ShardCache:
    def __init__(self, cache_dir: Path, cache_gb: float):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(cache_gb * (1024**3))
        self.lru = OrderedDict()

    def _current_bytes(self):
        return sum(self.lru.values())

    def _evict_if_needed(self):
        while self._current_bytes() > self.max_bytes and self.lru:
            name, _ = self.lru.popitem(last=False)
            try:
                (self.cache_dir / name).unlink()
            except FileNotFoundError:
                pass

    def ensure(self, s3_uri: str) -> Path:
        name = Path(parse_s3_uri(s3_uri)[1]).name
        local = self.cache_dir / name
        if local.exists():
            return local
        s3_download(s3_uri, local)
        self.lru[name] = get_file_size_bytes(local)
        self._evict_if_needed()
        return local


# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(path: Path, model, optimizer, scaler, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        path,
    )


def append_loss_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--s3_latents_prefix", type=str, required=True)
    ap.add_argument("--max_shards", type=int, default=0)

    ap.add_argument("--cache_dir", type=str, default="data/_s3_cache_latents")
    ap.add_argument("--cache_gb", type=float, default=50)

    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=1000)

    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--loss_csv", type=str, default="checkpoints/unet/loss.csv")

    ap.add_argument("--amp", action="store_true")

    args = ap.parse_args()

    device = pick_device()
    print("Device:", device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # -----------------------------
    # SHARD DISCOVERY (ALL cc12m_256_0000*)
    # -----------------------------
    lat_bucket, lat_prefix = parse_s3_uri(args.s3_latents_prefix)

    dataset_prefixes = s3_list_common_prefixes(lat_bucket, lat_prefix)
    if not dataset_prefixes:
        raise RuntimeError(f"No dataset folders under {args.s3_latents_prefix}")

    print(f"Found {len(dataset_prefixes)} dataset folders")

    shard_keys = []
    for dp in dataset_prefixes:
        shard_keys.extend(s3_list_keys(lat_bucket, dp, suffix=".pt"))

    shard_keys = sorted(set(shard_keys))
    random.shuffle(shard_keys)

    if args.max_shards > 0:
        shard_keys = shard_keys[: args.max_shards]

    if not shard_keys:
        raise RuntimeError("No .pt shards found in datasets")

    shard_uris = [f"s3://{lat_bucket}/{k}" for k in shard_keys]
    print(f"Total shards collected: {len(shard_uris)}")

    # -----------------------------
    # Model / optimizer
    # -----------------------------
    model = UNet().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    _, _, abar = make_beta_schedule(args.T, device=device)
    cache = ShardCache(Path(args.cache_dir), args.cache_gb)

    step = 0
    t0 = time.time()

    while step < args.steps:
        shard_uri = shard_uris[step % len(shard_uris)]
        local = cache.ensure(shard_uri)

        ckpt = torch.load(local, map_location="cpu")
        latents = ckpt["latents"]
        if not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents)
        latents = latents.contiguous()

        idx = torch.randint(0, latents.shape[0], (args.batch_size,))
        x0 = latents[idx].to(device)

        t = torch.randint(0, args.T, (args.batch_size,), device=device)
        eps = torch.randn_like(x0)
        abar_t = abar[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(abar_t) * x0 + torch.sqrt(1 - abar_t) * eps

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast("cuda"):
                pred = model(xt, t)
                loss = F.mse_loss(pred, eps)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(xt, t)
            loss = F.mse_loss(pred, eps)
            loss.backward()
            optimizer.step()

        step += 1

        if step % args.print_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {step:07d} | loss {loss.item():.6f} | {dt:.1f}s")

        if step % args.print_every == 0:
            append_loss_csv(
                Path(args.loss_csv),
                {"step": step, "loss": float(loss.item())},
            )

        if step % args.save_every == 0:
            save_checkpoint(
                Path("checkpoints/unet/unet_latest.pt"),
                model,
                optimizer,
                scaler,
                step,
            )


if __name__ == "__main__":
    main()
