import argparse
from pathlib import Path
import torch

from diffusion.schedule import DiffusionSchedule
from diffusion.sampler_gif import sample_ddpm_with_frames
from models.unet_mnist import TinyUNetCFG
from utils.io import load_checkpoint, save_grid
from utils.gif import save_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--tdim", type=int, default=128)
    parser.add_argument("--digit", type=int, default=7)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--every", type=int, default=10, help="save a GIF frame every N steps")
    parser.add_argument("--out_png", type=str, default="runs/mnist_cfg/samples/final.png")
    parser.add_argument("--out_gif", type=str, default="runs/mnist_cfg/samples/denoise.gif")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    ckpt_path = PROJECT_ROOT / args.ckpt
    out_png = PROJECT_ROOT / args.out_png
    out_gif = PROJECT_ROOT / args.out_gif

    model = TinyUNetCFG(base=args.base, tdim=args.tdim).to(device)
    model = load_checkpoint(model, ckpt_path, map_location=device)

    schedule = DiffusionSchedule(T=args.T, device=device)

    y = torch.full((args.n,), args.digit, device=device, dtype=torch.long)

    samples, frames = sample_ddpm_with_frames(
        schedule,
        model,
        n=args.n,
        y=y,
        cfg_scale=args.cfg_scale,
        device=device,
        every=args.every,
    )

    # Save final grid (all n samples)
    save_grid(samples, out_png, nrow=4)

    # Save GIF (only shows the first sample evolving)
    save_gif(frames, out_gif, fps=args.fps)

    print(f"saved final PNG: {out_png}")
    print(f"saved GIF: {out_gif}")


if __name__ == "__main__":
    main()
