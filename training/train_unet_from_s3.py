#!/usr/bin/env python3
import argparse
import time
import csv
import sys
import re
from pathlib import Path
from collections import OrderedDict
import random
from datetime import datetime
from typing import Optional

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
    bucket, _, key = rest.partition("/")
    return bucket, key


def s3_exists(s3_uri: str) -> bool:
    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def s3_download(s3_uri: str, local_path: Path):
    bucket, key = parse_s3_uri(s3_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(local_path))


def s3_upload(local_path: Path, s3_uri: str):
    bucket, key = parse_s3_uri(s3_uri)
    boto3.client("s3").upload_file(str(local_path), bucket, key)


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
    List immediate subfolders under a prefix (Delimiter='/').
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    prefixes = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            prefixes.append(p["Prefix"])
    return prefixes


# -----------------------------
# Run directory helpers
# -----------------------------
def prepare_local_run_dir_ordered(base_dir: str = "training", run_name: Optional[str] = None):
    """
    Creates:
      training/training_MM_DD_YYYY_run_XX/
        checkpoints/
        logs/
        samples/   (empty until you generate samples)

    If run_name is provided, uses that exactly (no numbering).
    Otherwise auto-increments XX per day.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        date_tag = datetime.now().strftime("%m_%d_%Y")
        pattern = re.compile(rf"training_{date_tag}_run_(\d+)$")
        existing = []
        for p in base.iterdir():
            if p.is_dir():
                m = pattern.match(p.name)
                if m:
                    existing.append(int(m.group(1)))
        next_run = max(existing, default=0) + 1
        run_name = f"training_{date_tag}_run_{next_run:02d}"

    run_dir = base / run_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    sample_dir = run_dir / "samples"

    # exist_ok=False so we never silently overwrite a run folder
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    sample_dir.mkdir(parents=True, exist_ok=False)

    return run_dir, ckpt_dir, log_dir, sample_dir


def pull_s3_artifacts_into_run(
    *,
    s3_ckpt_uri: str,
    s3_loss_uri: str,
    local_ckpt_path: Path,
    local_loss_path: Path,
    overwrite: bool = False,
):
    """
    Pulls S3 artifacts into the local run folder.
    - Checkpoint: downloads if exists (or leaves missing if none on S3)
    - loss.csv: downloads if exists (or will be created as training logs)
    - samples folder stays empty (created by run dir helper)
    """
    # checkpoint
    if (not local_ckpt_path.exists()) or overwrite:
        if s3_ckpt_uri and s3_exists(s3_ckpt_uri):
            print(f"[S3] Download checkpoint -> {local_ckpt_path}")
            s3_download(s3_ckpt_uri, local_ckpt_path)
        else:
            print("[S3] No checkpoint found to download (starting fresh).")

    # loss.csv
    if (not local_loss_path.exists()) or overwrite:
        if s3_loss_uri and s3_exists(s3_loss_uri):
            print(f"[S3] Download loss.csv -> {local_loss_path}")
            s3_download(s3_loss_uri, local_loss_path)
        else:
            print("[S3] No loss.csv found to download (will create locally).")


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
            # refresh LRU
            self.lru.pop(name, None)
            self.lru[name] = get_file_size_bytes(local)
            return local

        s3_download(s3_uri, local)
        self.lru[name] = get_file_size_bytes(local)
        self._evict_if_needed()
        return local


# -----------------------------
# Checkpoint + logging helpers
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


def load_checkpoint_into(model, optimizer, scaler, ckpt_path: Path, use_amp: bool):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # support raw state_dict
        model.load_state_dict(ckpt)

    if isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if use_amp and scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print("[WARN] Could not load scaler state, continuing without it:", e)

    step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
    return step


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--s3_latents_prefix", type=str, required=True)
    ap.add_argument("--max_shards", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default="data/_s3_cache_latents")
    ap.add_argument("--cache_gb", type=float, default=50)

    # Training control
    # NOTE: --steps means "how many steps to run THIS session"
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=1000)

    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--upload_every", type=int, default=1000, help="0 disables uploads during run")

    ap.add_argument("--amp", action="store_true")

    # Local run folder layout
    ap.add_argument("--run_base_dir", type=str, default="training")
    ap.add_argument("--run_name", type=str, default=None, help="Optional fixed run name; otherwise auto run_XX")

    # Resume + S3 artifacts (defaults match your bucket)
    ap.add_argument(
        "--s3_ckpt_uri",
        type=str,
        default="s3://danielumarov-diffusion-data/checkpoints/unet/unet_latest.pt",
        help="Global S3 latest checkpoint (read + write).",
    )
    ap.add_argument(
        "--s3_loss_uri",
        type=str,
        default="s3://danielumarov-diffusion-data/checkpoints/unet/loss.csv",
        help="Global S3 loss.csv (read optional + write).",
    )
    ap.add_argument("--resume", action="store_true", help="Resume from S3 latest if it exists")

    args = ap.parse_args()

    # -----------------------------
    # Create new ordered run folder for TODAY
    # -----------------------------
    run_dir, ckpt_dir, log_dir, sample_dir = prepare_local_run_dir_ordered(
        base_dir=args.run_base_dir,
        run_name=args.run_name,
    )
    print("Run dir:", run_dir)
    print("Checkpoints dir:", ckpt_dir)
    print("Logs dir:", log_dir)
    print("Samples dir (starts empty):", sample_dir)

    local_ckpt_path = ckpt_dir / "unet_latest.pt"
    local_loss_path = log_dir / "loss.csv"

    # Pull S3 latest into this new run folder (if present)
    pull_s3_artifacts_into_run(
        s3_ckpt_uri=args.s3_ckpt_uri,
        s3_loss_uri=args.s3_loss_uri,
        local_ckpt_path=local_ckpt_path,
        local_loss_path=local_loss_path,
        overwrite=False,
    )

    # -----------------------------
    # Device / AMP
    # -----------------------------
    device = pick_device()
    print("Device:", device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # -----------------------------
    # Shard discovery (find all dataset folders + all .pt shards)
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

    # -----------------------------
    # Resume: load from the checkpoint we just pulled into THIS run folder
    # -----------------------------
    step = 0
    if args.resume:
        if local_ckpt_path.exists():
            step = load_checkpoint_into(model, optimizer, scaler, local_ckpt_path, use_amp=use_amp)
            print(f"[RESUME] Loaded checkpoint from: {local_ckpt_path}")
            print(f"[RESUME] Source S3 key: {args.s3_ckpt_uri}")
            print(f"[RESUME] Starting at step: {step}")
        else:
            print("[RESUME] Requested, but no checkpoint present in this run folder. Starting fresh.")

    # -----------------------------
    # Training loop (runs args.steps steps THIS session)
    # -----------------------------
    start_step = step
    end_step = start_step + args.steps
    t0 = time.time()

    while step < end_step:
        shard_uri = shard_uris[step % len(shard_uris)]
        local_shard = cache.ensure(shard_uri)

        data = torch.load(local_shard, map_location="cpu")
        latents = data["latents"]
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
            append_loss_csv(local_loss_path, {"step": step, "loss": float(loss.item())})

        if step % args.save_every == 0:
            save_checkpoint(local_ckpt_path, model, optimizer, scaler, step)

        if args.upload_every > 0 and (step % args.upload_every == 0):
            # upload latest weights + loss to global S3 keys
            if args.s3_ckpt_uri:
                print(f"[S3] Upload checkpoint -> {args.s3_ckpt_uri}")
                s3_upload(local_ckpt_path, args.s3_ckpt_uri)
            if args.s3_loss_uri and local_loss_path.exists():
                print(f"[S3] Upload loss.csv -> {args.s3_loss_uri}")
                s3_upload(local_loss_path, args.s3_loss_uri)

    # Final save + upload (so you don't lose last few steps)
    save_checkpoint(local_ckpt_path, model, optimizer, scaler, step)

    if args.s3_ckpt_uri:
        print(f"[S3] Final upload checkpoint -> {args.s3_ckpt_uri}")
        s3_upload(local_ckpt_path, args.s3_ckpt_uri)

    if args.s3_loss_uri and local_loss_path.exists():
        print(f"[S3] Final upload loss.csv -> {args.s3_loss_uri}")
        s3_upload(local_loss_path, args.s3_loss_uri)

    print("Done.")
    print("Local run folder:", run_dir)
    print("Local latest checkpoint:", local_ckpt_path)
    print("Local loss log:", local_loss_path)


if __name__ == "__main__":
    main()
