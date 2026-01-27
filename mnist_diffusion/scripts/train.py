import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mnist_diffusion.data.mnist import get_mnist_dataloader
from mnist_diffusion.diffusion.schedule import DiffusionSchedule
from mnist_diffusion.diffusion.sampler import sample_ddpm
from mnist_diffusion.models.unet_mnist import TinyUNetCFG
from mnist_diffusion.utils.io import save_checkpoint, save_grid
from mnist_diffusion.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--tdim", type=int, default=128)
    parser.add_argument("--p_uncond", type=float, default=0.1, help="Label drop prob for CFG training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="runs/mnist_cfg")
    args = parser.parse_args()

    # Anchor outputs inside mnist_diffusion/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUTDIR = PROJECT_ROOT / args.outdir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    dl = get_mnist_dataloader(batch_size=args.batch_size, train=True)

    model = TinyUNetCFG(base=args.base, tdim=args.tdim).to(device)
    schedule = DiffusionSchedule(T=args.T, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        for i, (x0, y) in enumerate(dl):
            x0 = x0.to(device)
            y = y.to(device)

            t = torch.randint(0, schedule.T, (x0.size(0),), device=device, dtype=torch.long)
            xt, noise = schedule.q_sample(x0, t)

            # CFG training: randomly drop labels to null class
            drop = (torch.rand(x0.size(0), device=device) < args.p_uncond)
            y_in = y.clone()
            y_in[drop] = model.null_class

            noise_pred = model(xt, t, y_in)
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if i % 200 == 0:
                print(f"epoch {ep} iter {i} loss {loss.item():.4f}")

        # Save checkpoint each epoch
        ckpt_path = OUTDIR / "checkpoints" / f"ep{ep}.pt"
        save_checkpoint(model, ckpt_path)

        # Sample a grid: digits 0..9 repeated
        labels = torch.tensor([0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5], device=device)
        samples = sample_ddpm(schedule, model, n=labels.size(0), y=labels, cfg_scale=3.0, device=device)
        save_grid(samples, OUTDIR / "samples" / f"ep{ep}_cfg3.png", nrow=4)

        print(f"saved: {ckpt_path}")

    print("done.")


if __name__ == "__main__":
    main()
