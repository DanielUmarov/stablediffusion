import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_emb = nn.Linear(time_dim, out_ch)

        self.residual = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        x = self.block(x, t)
        return x, self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block = ResBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x, skip, t):
        x = self.up(x)                 # upsample FIRST
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=4, base=64, time_dim=256):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.init = nn.Conv2d(in_channels, base, 3, padding=1)

        self.down1 = Down(base, base * 2, time_dim)       # -> skip is base*2, x is base*2 at half res
        self.down2 = Down(base * 2, base * 4, time_dim)   # -> skip is base*4, x is base*4 at quarter res

        self.mid = ResBlock(base * 4, base * 4, time_dim)

        # NOTE: (in_ch, skip_ch, out_ch)
        self.up1 = Up(base * 4, base * 4, base * 2, time_dim)  # quarter->half, concat with s2
        self.up2 = Up(base * 2, base * 2, base, time_dim)      # half->full, concat with s1

        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t = self.time_embed(t)

        x = self.init(x)

        s1, x = self.down1(x, t)   # s1: (B, 2b, 32,32)   x: (B,2b,16,16)
        s2, x = self.down2(x, t)   # s2: (B, 4b, 16,16)   x: (B,4b, 8, 8)

        x = self.mid(x, t)         # (B,4b,8,8)

        x = self.up1(x, s2, t)     # up to (B,2b,16,16), concat skip (4b) inside block
        x = self.up2(x, s1, t)     # up to (B,b,32,32)

        return self.out(x)

if __name__ == "__main__":
    model = UNet(in_channels=4)
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = model(x, t)
    print("out:", y.shape)

