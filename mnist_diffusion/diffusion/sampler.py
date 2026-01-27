import torch


@torch.no_grad()
def p_sample_ddpm(schedule, model, x, t_int: int, y, *, cfg_scale: float = 0.0):
    """
    One reverse step x_t -> x_{t-1}.
    If cfg_scale > 0, uses classifier-free guidance (CFG).
    """
    device = x.device
    b = x.size(0)
    t = torch.full((b,), t_int, device=device, dtype=torch.long)

    beta_t = schedule.betas[t_int]
    alpha_t = schedule.alphas[t_int]
    ab_t = schedule.alphas_bar[t_int]

    if cfg_scale and cfg_scale > 0:
        # One forward via batch concat for (uncond, cond)
        y_null = torch.full((b,), model.null_class, device=device, dtype=torch.long)

        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y_null, y], dim=0)

        eps = model(x_in, t_in, y_in)
        eps_uncond, eps_cond = eps.chunk(2, dim=0)
        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    else:
        eps_pred = model(x, t, y)

    # DDPM mean (noise prediction form)
    mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - ab_t)) * eps_pred)

    if t_int == 0:
        return mean

    noise = torch.randn_like(x)
    return mean + torch.sqrt(beta_t) * noise


@torch.no_grad()
def sample_ddpm(schedule, model, n: int, shape=(1, 28, 28), y=None, cfg_scale: float = 0.0, device="cpu"):
    """
    Generate n samples. If y is provided, it must be (n,) long tensor of labels 0..9.
    """
    model.eval()
    x = torch.randn((n, *shape), device=device)

    if y is not None:
        y = y.to(device=device, dtype=torch.long)

    for t in reversed(range(schedule.T)):
        x = p_sample_ddpm(schedule, model, x, t, y, cfg_scale=cfg_scale)

    return x
