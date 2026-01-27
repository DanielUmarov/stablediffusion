import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# ----------------------------
# Utilities: time embedding
# ----------------------------
def sinusoidal_timestep_embedding(timesteps, dim, max_period=10000):
    """
    timesteps: (B,) int64
    returns: (B, dim) float
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# ----------------------------
# Tiny UNet-ish model for 28x28
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.tproj = nn.Linear(tdim, ch)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.tproj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class TinyUNetCond(nn.Module):
    def __init__(self, in_ch=1, base=64, tdim=128, num_classes=10):
        super().__init__()
        self.tdim = tdim

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        # class embedding (digit 0-9)
        self.class_emb = nn.Embedding(num_classes, tdim)

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.rb1 = ResBlock(base, tdim)
        self.down1 = Down(base)

        self.rb2 = ResBlock(base, tdim)
        self.down2 = Down(base)

        self.mid = ResBlock(base, tdim)

        self.up2 = Up(base)
        self.rb3 = ResBlock(base, tdim)

        self.up1 = Up(base)
        self.rb4 = ResBlock(base, tdim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t, y):
        """
        x: (B,1,28,28)
        t: (B,) int64
        y: (B,) int64 labels in [0..9]
        """
        temb = sinusoidal_timestep_embedding(t, self.tdim)
        temb = self.time_mlp(temb)

        # add class info into the same embedding space
        cemb = self.class_emb(y)
        temb = temb + cemb

        x0 = self.in_conv(x)

        h1 = self.rb1(x0, temb)
        d1 = self.down1(h1)

        h2 = self.rb2(d1, temb)
        d2 = self.down2(h2)

        m = self.mid(d2, temb)

        u2 = self.up2(m)
        u2 = u2 + h2
        u2 = self.rb3(u2, temb)

        u1 = self.up1(u2)
        u1 = u1 + h1
        u1 = self.rb4(u1, temb)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out


# ----------------------------
# Diffusion schedule + helpers
# ----------------------------
class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar

        self.sqrt_ab = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_ab = torch.sqrt(1.0 - alphas_bar)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        # gather per-batch scalars
        sqrt_ab_t = self.sqrt_ab[t][:, None, None, None]
        sqrt_omab_t = self.sqrt_one_minus_ab[t][:, None, None, None]
        return sqrt_ab_t * x0 + sqrt_omab_t * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t):
        """
        One reverse step: x_t -> x_{t-1}
        """
        b = x.size(0)
        t_batch = torch.full((b,), t, device=self.device, dtype=torch.long)

        eps_pred = model(x, t_batch)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        ab_t = self.alphas_bar[t]

        # DDPM mean (predict noise)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - ab_t)

        mean = coef1 * (x - coef2 * eps_pred)

        if t == 0:
            return mean
        noise = torch.randn_like(x)
        var = beta_t
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, n=16, shape=(1, 28, 28)):
        model.eval()
        x = torch.randn((n, *shape), device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x

# ----------------------------
# Train + sample
# ----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # MNIST: map to [-1, 1]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

    model = TinyUNet().to(device)
    diff = Diffusion(T=200, device=device)  # 200 steps is faster for MNIST

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    epochs = 3
    for ep in range(epochs):
        model.train()
        for i, (x0, _) in enumerate(dl):
            x0 = x0.to(device)

            t = torch.randint(0, diff.T, (x0.size(0),), device=device, dtype=torch.long)
            xt, noise = diff.q_sample(x0, t)

            noise_pred = model(xt, t)
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if i % 200 == 0:
                print(f"ep {ep} iter {i} loss {loss.item():.4f}")

        # sample each epoch
        samples = diff.sample(model, n=16)
        # back to [0,1] for saving
        grid = utils.make_grid((samples.clamp(-1, 1) + 1) / 2, nrow=4)
        utils.save_image(grid, f"mnist_samples_ep{ep}.png")
        print(f"saved mnist_samples_ep{ep}.png")

    torch.save(model.state_dict(), "mnist_diffusion.pt")
    print("saved mnist_diffusion.pt")

if __name__ == "__main__":
    main()
