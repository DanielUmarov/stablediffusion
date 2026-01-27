import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    timesteps: (B,) int64
    returns: (B, dim) float32
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


class ResBlock(nn.Module):
    def __init__(self, ch: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.tproj = nn.Linear(tdim, ch)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.tproj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class TinyUNetCFG(nn.Module):
    """
    MNIST-sized UNet-ish backbone with classifier-free guidance (CFG) support via a null class embedding.
    """
    def __init__(self, in_ch: int = 1, base: int = 64, tdim: int = 128, num_classes: int = 10):
        super().__init__()
        self.tdim = tdim
        self.num_classes = num_classes
        self.null_class = num_classes  # 10 is the "no label" token

        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        # +1 for null class
        self.class_emb = nn.Embedding(num_classes + 1, tdim)

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

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        """
        x: (B,1,28,28)
        t: (B,) int64
        y:
          - (B,) labels 0..9
          - OR None for unconditional
          - OR (B,) where some entries are null_class
        """
        temb = sinusoidal_timestep_embedding(t, self.tdim)
        temb = self.time_mlp(temb)

        if y is None:
            y = torch.full((x.size(0),), self.null_class, device=x.device, dtype=torch.long)
        else:
            y = y.to(device=x.device, dtype=torch.long)

        temb = temb + self.class_emb(y)

        h = self.in_conv(x)

        h1 = self.rb1(h, temb)
        d1 = self.down1(h1)

        h2 = self.rb2(d1, temb)
        d2 = self.down2(h2)

        m = self.mid(d2, temb)

        u2 = self.up2(m) + h2
        u2 = self.rb3(u2, temb)

        u1 = self.up1(u2) + h1
        u1 = self.rb4(u1, temb)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out  # predicted noise eps
