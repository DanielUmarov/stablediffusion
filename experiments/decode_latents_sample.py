#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKL


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data/latents/_recon")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()
    shard_path = Path(args.shard)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(shard_path, map_location="cpu")
    latents = ckpt["latents"][: args.n]
    latent_scale = ckpt.get("latent_scale", 0.18215)
    vae_id = ckpt["vae_id"]
    image_size = ckpt["image_size"]

    vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae").eval().to(device)
    vae = vae.to(dtype=latents.dtype)

    # unscale
    z = latents.to(device) / latent_scale
    # decode -> [-1,1]
    x = vae.decode(z).sample
    # [-1,1] -> [0,1]
    x = (x + 1) / 2.0
    x = x.clamp(0, 1)

    for i in range(x.shape[0]):
        img = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img).save(out_dir / f"recon_{i:02d}_{image_size}.png")

    print("Wrote recon images to", out_dir)


if __name__ == "__main__":
    main()
