import torch


class DiffusionSchedule:
    def __init__(self, T: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        self.T = int(T)
        self.device = device

        betas = torch.linspace(beta_start, beta_end, self.T, device=device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar

        self.sqrt_ab = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_ab = torch.sqrt(1.0 - alphas_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """
        x0: (B,1,28,28)
        t: (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab_t = self.sqrt_ab[t][:, None, None, None]
        sqrt_omab_t = self.sqrt_one_minus_ab[t][:, None, None, None]
        xt = sqrt_ab_t * x0 + sqrt_omab_t * noise
        return xt, noise
