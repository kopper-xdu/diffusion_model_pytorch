import torch


def linear_schedule(start, end, time_steps):
    return torch.linspace(start, end, time_steps)


time_steps = 1000

betas = linear_schedule(1e-4, 2e-2, time_steps)

alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)
one_minus_alphas_cumprod_sqrt = torch.sqrt(1 - alphas_cumprod)


def q_sample(x_0, t, noise=None):
    device = x_0.device
    if noise is None:
        noise = torch.randn(x_0.shape, device=device)
    mean = (alphas_cumprod_sqrt[t].unsqueeze(1).to(device) * x_0.reshape(x_0.shape[0], -1)).reshape(x_0.shape)
    std = one_minus_alphas_cumprod_sqrt[t].unsqueeze(1)
    return mean + (std.to(device) * noise.reshape(noise.shape[0], -1)).reshape(noise.shape)


sqrt_recip_alphas = torch.sqrt(1. / alphas)
alphas_cumprod_prev = torch.cat((torch.tensor([1.]), alphas_cumprod[:-1]), 0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
log_posterior_variance = torch.log(posterior_variance.clamp(min=1e-20))  # 数值稳定


def p_sample(model, x, t):
    device = x.device
    with torch.no_grad():
        noise = torch.randn(x.shape, device=device) if t != 0 else torch.zeros(x.shape, device=device)
        mean = sqrt_recip_alphas[t].to(device) * (
                    x - betas[t].to(device) / one_minus_alphas_cumprod_sqrt[t].to(device) * model(x, t))
        logvar = log_posterior_variance[t].to(device)
    return mean + torch.exp(0.5 * logvar) * noise


def p_sample_loop(model, x, clip_denoised=False):  # TODO
    """Sample from p_{theta} (x_{0:t-1} | x_t)"""
    device = next(model.parameters()).device
    x = x.to(device)
    res = []
    for i in reversed(range(time_steps)):
        x = p_sample(model, x, torch.tensor([i], device=device))
        res.append(x)
    if clip_denoised:
        res = [torch.clip(x, -1, 1) for x in res]
    return res
