import torch
import torch.nn.functional as F
import numpy as np
import tqdm

def sampler(x, model, sde, device, W, eps, dataset, steps=1000, sampler_input=None):
    """
    Sampler for 1D data using SDE-based generation
    
    Args:
        x: Initial noise tensor
        model: Score model
        sde: SDE object (VPSDE1D)
        device: Device to run on
        W: Noise generator (HilbertNoise)
        eps: Small epsilon value for numerical stability
        dataset: Dataset name (for gradient clipping)
        steps: Number of sampling steps
        sampler_input: Optional pre-generated noise
    
    Returns:
        Generated samples
    """
    def sde_score_update(x, s, t):
        """
        input: x_s, s, t
        output: x_t
        """
        models = model(x, s)
        score_s = models * torch.pow(sde.marginal_std(s), -(2.0 - 1))[:, None].to(device)

        beta_step = sde.beta(s) * (s - t)
        x_coeff = 1 + beta_step / 2.0

        noise_coeff = torch.pow(beta_step, 1 / 2.0)
        if sampler_input == None:
            e = W.sample(x.shape)
        else:
            e = W.free_sample(free_input=sampler_input)

        score_coeff = beta_step
        x_t = x_coeff[:, None].to(device) * x + score_coeff[:, None].to(device) * score_s + noise_coeff[:, None].to(device) * e.to(device)

        return x_t

    timesteps = torch.linspace(sde.T, eps, steps + 1).to(device)

    with torch.no_grad():
        for i in tqdm.tqdm(range(steps)):
            vec_s = torch.ones((x.shape[0],)).to(device) * timesteps[i]
            vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]

            x = sde_score_update(x, vec_s, vec_t)

            # Gradient clipping for stability
            size = x.shape
            l = x.shape[0]
            x = x.reshape((l, -1))
            indices = x.norm(dim=1) > 10
            if dataset == 'Gridwatch':
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 17
            else:
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 10
            x = x.reshape(size)

    return x