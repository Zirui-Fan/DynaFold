import os
import re
import tqdm
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def edm_sampler(
    net, x, cond_x, aa_types, sequences,
    condition_mask=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.01, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sigma_data = 0.5
):
    B,T = x.shape[:2]
    x_next = x.clone()
    # Time step discretization.
    step_indices = torch.arange(num_steps, device=x.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    if condition_mask is None:
        condition_mask = torch.zeros((B,T),dtype=torch.bool,device=x.device)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        t_cur = t_cur.reshape(-1,1,1,1)
        t_next = t_next.reshape(-1,1,1,1)

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        c_in = 1 / (sigma_data ** 2 + t_hat ** 2).sqrt()
        denoised = net(c_in*x_hat, cond_x, t_hat.log()/4, aa_types, sequences, condition_mask=condition_mask)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            c_in_next =  1 / (sigma_data ** 2 + t_next ** 2).sqrt()
            denoised = net(c_in_next*x_next, cond_x, t_next.log()/4, aa_types, sequences, condition_mask=condition_mask)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    else:
        return x_next