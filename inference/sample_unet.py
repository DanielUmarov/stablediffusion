#!/usr/bin/env python3
import sys
from pathlib import Path

# -------------------------------------------------
# Add repo root to Python path
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import torch
from diffusers import AutoencoderKL
from PIL import Image

from models.unet.unet import UNet


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


# -------------------------------------------------
# DDIM sampler (epsilon prediction)
# -------------------------------------------------
@torch.no_grad()
def ddim_sample(
    model,
    shape,
    alphas_cumprod,
    steps=50,
    eta=0.0,
    device="cpu",
):
    """
    model predicts epsilon
    """
    T = alphas_cumprod.shape[0]

    # timesteps on correct device
    step_ids = torch.linspace(T - 1, 0, steps, device=device).long()

    # start from pure noise
    x = torch.randn(shape, device=device, dtype=torch.float32)

    for i in range(len(step_ids) - 1):
        t = step_ids[i]
        t_prev = step_ids[i + 1]

        # IMPORTANT: force these onto device (MPS bug)
        a_t = alphas_cumprod[t].to(device)
        a_prev = alphas_cumprod[t_prev].to(device)

        # batch timesteps
        t_batch = torch.full(
            (x.shape[0],),
            int(t.item()),
            device=device,
            dtype=torch.long,
        )

        eps = model(x, t_batch)

        # predict x0
        x0 = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)

        # DDIM update
        if eta > 0:
            sigma = (
                eta
                * torch.sqrt((1 - a_prev) / (1 - a_t))
                * torch.sqrt(1 - a_t / a_prev)
            )
            noise = torch.randn_like(x)
        else:
            sigma = 0.0
            noise = 0.0

        x = (
            torch.sqrt(a_prev) * x0
            + torch.sqrt(1.0 - a_prev - sigma ** 2) * eps
            + sigma * noise
        )

    return x


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--vae_id", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--out_dir", type=str, default="samples")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()

    device = pick_device()
    print("Using device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load UNet
    # -------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location="cpu")

    model = UNet()
    model.load_state_dict(ckpt["model"])
    model = model.to(device=device, dtype=torch.float32).eval()

    # -------------------------------------------------
    # Noise schedule
    # -------------------------------------------------
    T = 1000
    betas = make_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # -------------------------------------------------
    # Sample latents
    # -------------------------------------------------
    latent_shape = (
        args.n,
        4,
        args.image_size // 8,
        args.image_size // 8,
    )

    latents = ddim_sample(
        model=model,
        shape=latent_shape,
        alphas_cumprod=alphas_cumprod,
        steps=args.steps,
        eta=0.0,
        device=device,
    )

    latents = latents.to(dtype=torch.float32)

    # -------------------------------------------------
    # Decode with VAE
    # -------------------------------------------------
    vae = AutoencoderKL.from_pretrained(
        args.vae_id,
        subfolder="vae",
    )
    vae = vae.to(device=device, dtype=torch.float32).eval()

    latent_scale = 0.18215
    z = latents / latent_scale

    imgs = vae.decode(z).sample
    imgs = (imgs + 1.0) / 2.0
    imgs = imgs.clamp(0.0, 1.0)

    for i in range(imgs.shape[0]):
        img = (
            imgs[i]
            .detach()          # IMPORTANT
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            * 255
        ).astype("uint8")

        Image.fromarray(img).save(out_dir / f"sample_{i:02d}.png")

    print("Saved samples to:", out_dir)


if __name__ == "__main__":
    main()
