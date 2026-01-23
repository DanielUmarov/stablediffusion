#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import Counter

import torch
import pandas as pd
import matplotlib.pyplot as plt


LOSS_RE = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|\s+shard\s+(\S+)")


def parse_log(log_path: Path) -> pd.DataFrame:
    rows = []
    text = log_path.read_text(errors="ignore").splitlines()
    for line in text:
        m = LOSS_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        loss = float(m.group(2))
        shard = m.group(3)
        rows.append({"step": step, "loss": loss, "shard": shard})
    if not rows:
        raise ValueError(f"No loss lines found in {log_path}. Make sure it contains: 'step #### | loss ... | shard ...'")
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def add_ema(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    # ema_t = alpha*ema_{t-1} + (1-alpha)*x_t
    ema = []
    v = None
    for x in df["loss"].tolist():
        v = x if v is None else alpha * v + (1 - alpha) * x
        ema.append(v)
    out = df.copy()
    out["ema"] = ema
    return out


def plot_loss(df: pd.DataFrame, out_path: Path | None = None, show: bool = True):
    plt.figure()
    plt.plot(df["step"], df["loss"], label="loss")
    if "ema" in df.columns:
        plt.plot(df["step"], df["ema"], label="ema")
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        print("Saved plot:", out_path)
    if show:
        plt.show()
    else:
        plt.close()


def shard_coverage(df: pd.DataFrame, topk: int = 20):
    c = Counter(df["shard"].tolist())
    total = sum(c.values())
    items = c.most_common(topk)
    print("\nShard coverage (from log):")
    for shard, n in items:
        print(f"  {shard:20s}  {n:6d}  ({100*n/total:5.1f}%)")
    if len(c) > topk:
        print(f"  ... ({len(c) - topk} more shards not shown)")
    print(f"Total logged steps: {total}")


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def get_model_state(ckpt: dict) -> dict:
    # supports {"model": state_dict} format used in your trainer
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    # fallback: sometimes people save raw state dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError("Could not find model state_dict in checkpoint. Expected key 'model'.")


def compare_weights(ckpt_a: Path, ckpt_b: Path):
    a = load_checkpoint(ckpt_a)
    b = load_checkpoint(ckpt_b)
    sa = get_model_state(a)
    sb = get_model_state(b)

    keys_a = set(sa.keys())
    keys_b = set(sb.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    if only_a or only_b:
        print("\nWarning: checkpoint param key mismatch.")
        if only_a:
            print(f"  Only in A ({ckpt_a.name}): {len(only_a)} keys")
        if only_b:
            print(f"  Only in B ({ckpt_b.name}): {len(only_b)} keys")

    # Compute norms of diffs
    total_l2 = 0.0
    total_linf = 0.0
    total_params = 0

    per_layer = []
    for k in common:
        ta = sa[k]
        tb = sb[k]
        if not (isinstance(ta, torch.Tensor) and isinstance(tb, torch.Tensor)):
            continue
        if ta.shape != tb.shape:
            continue
        da = (ta.float() - tb.float())
        l2 = torch.norm(da).item()
        linf = torch.max(torch.abs(da)).item()
        n = da.numel()
        total_l2 += l2
        total_linf = max(total_linf, linf)
        total_params += n
        per_layer.append((k, l2, linf, n))

    per_layer.sort(key=lambda x: x[1], reverse=True)

    print("\nWeight-change check:")
    print(f"  A: {ckpt_a}")
    print(f"  B: {ckpt_b}")
    print(f"  Compared layers: {len(per_layer)} | Total params compared: {total_params:,}")
    print(f"  Sum layer L2 norms: {total_l2:.6f} | Max abs diff (global Linf): {total_linf:.6f}")

    print("\nTop 10 layers by L2 diff:")
    for k, l2, linf, n in per_layer[:10]:
        print(f"  {k:50s}  L2={l2:.6f}  Linf={linf:.6f}  n={n:,}")


def find_checkpoints(ckpt_dir: Path):
    cks = sorted(ckpt_dir.glob("unet_step_*.pt"))
    latest = ckpt_dir / "unet_latest.pt"
    return cks, latest if latest.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="", help="Path to training stdout log text file")
    ap.add_argument("--ema", type=float, default=0.98, help="EMA alpha (0.9-0.999 typical)")
    ap.add_argument("--plot_out", type=str, default="", help="Optional path to save plot PNG")
    ap.add_argument("--no_show", action="store_true", help="Do not pop up a window (still saves if --plot_out)")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints/unet", help="Directory with unet_step_*.pt")
    ap.add_argument("--ckpt_a", type=str, default="", help="Checkpoint A to compare")
    ap.add_argument("--ckpt_b", type=str, default="", help="Checkpoint B to compare")
    args = ap.parse_args()

    # --- Log analysis ---
    if args.log:
        log_path = Path(args.log)
        df = parse_log(log_path)
        df = add_ema(df, args.ema)

        print("\nLoss summary:")
        print(df[["step", "loss", "ema"]].describe())

        shard_coverage(df)

        out_path = Path(args.plot_out) if args.plot_out else None
        plot_loss(df, out_path=out_path, show=(not args.no_show))

    # --- Checkpoint analysis ---
    ckpt_dir = Path(args.ckpt_dir)
    cks, latest = find_checkpoints(ckpt_dir)

    if cks:
        print("\nFound checkpoints:")
        for p in cks[-10:]:
            step = load_checkpoint(p).get("step", None)
            print(f"  {p.name}  (step={step})")
        if len(cks) > 10:
            print(f"  ... ({len(cks)-10} older not shown)")
    if latest:
        print("\nFound:", latest)

    # Compare weights
    if args.ckpt_a and args.ckpt_b:
        compare_weights(Path(args.ckpt_a), Path(args.ckpt_b))
    else:
        # Auto-compare oldest vs newest if we have >=2
        if len(cks) >= 2:
            compare_weights(cks[0], cks[-1])


if __name__ == "__main__":
    main()
