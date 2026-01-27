import argparse
import torch
from pathlib import Path

from diffusion.schedule import DiffusionSchedule
from diffusion.sampler import sample_ddpm
from models.unet_mnist import TinyUNetCFG
from utils.io import load_checkpoint, save_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--tdim", type=int, default=128)
    parser.add_argument("--digit", type=int, default=7)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--out", type=str, default="runs/mnist_cfg/samples/sample.png")
    args = parser.parse_args()

    # --- device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- project-root–anchored paths ---
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    ckpt_path = PROJECT_ROOT / args.ckpt
    out_path = PROJECT_ROOT / args.out

    # --- model ---
    model = TinyUNetCFG(base=args.base, tdim=args.tdim).to(device)
    model = load_checkpoint(model, ckpt_path, map_location=device)

    # --- diffusion schedule ---
    schedule = DiffusionSchedule(T=args.T, device=device)

    # --- labels ---
    y = torch.full((args.n,), args.digit, device=device, dtype=torch.long)

    # --- sampling ---
    samples = sample_ddpm(
        schedule,
        model,
        n=args.n,
        y=y,
        cfg_scale=args.cfg_scale,
        device=device,
    )

    # --- save ---
    save_grid(samples, out_path, nrow=4)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
