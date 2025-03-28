# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch

from .util import unsqueeze_xdim

from ipdb import set_trace as debug
from i2sb.util import clear_color, clear
import matplotlib.pyplot as plt


def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class Diffusion():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()


    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False, verbose=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        if verbose:
            return xt_prev, mu_x0
        else:
            return xt_prev
     

    def i2sb_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)
        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='I2SB sampling', total=len(steps)-1) if verbose else pair_steps
        cnt = 0
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if mask is not None:
                
                # import matplotlib.pyplot as plt
                
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                # if cnt % 10 == 0:
                #     plt.subplot(1,2,1)
                #     plt.imshow(clear_color(xt))
                xt = (1. - mask) * xt_true + mask * xt
                #     plt.subplot(1,2,2)
                #     plt.imshow(clear_color(xt))
                #     plt.show()
                    
                #     import ipdb; ipdb.set_trace()

            cnt += 1
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    @torch.inference_mode()
    def i3sb_p_posterior(self, nprev, n, x_n, x_N, x0, ita):
        assert nprev < n
        std_n     = self.std_fwd[n]   #sigma_(n+1)
        std_nprev = self.std_fwd[nprev]   #sigma_(n)
        std_delta = (std_n**2 - std_nprev**2).sqrt() #alpha_n
        std_bar_nprev = self.std_bwd[nprev] #sigma_bar(n)
        std_bar_n = self.std_bwd[n] #sigma_bar(n+1)

        gn = std_nprev*std_delta/(((std_nprev**2)+(std_delta**2)).sqrt())
        gn = ita * gn
        
        gn_max = (((std_nprev**2)*(std_bar_nprev**2))/((std_nprev**2)+(std_bar_nprev**2))).sqrt()
        #gn = min(gn_max, gn)
        
        xt_prev = (std_bar_nprev**2)*x0/((std_bar_nprev**2)+(std_nprev**2))
        xt_prev = xt_prev + (std_nprev**2)*x_N/((std_bar_nprev**2)+(std_nprev**2))

        if gn<gn_max:
            k = (((std_nprev**2)*(std_bar_nprev**2))-((gn**2)*((std_nprev**2)+(std_bar_nprev**2)))).sqrt()/(std_n*std_bar_n)
            k = k * (((std_bar_n**2)+(std_n**2))/((std_nprev**2)+(std_bar_nprev**2))).sqrt()
            xt_prev = xt_prev + k * x_n

            xt_prev = xt_prev - k *(((std_bar_n**2)*x0/((std_bar_n**2)+(std_n**2)))+((std_n**2)*x_N/((std_bar_n**2)+(std_n**2))))
        else:
            gn = gn_max    
        
        #mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)
        #xt_prev2 = mu_x0 * x0 + mu_xn * x_n
        #r = xt_prev-xt_prev2
        #rTr = torch.sum(r*r, dim=[1,2,3])
        #print(rTr)
        if nprev>0:
            xt_prev = xt_prev + gn* torch.randn_like(xt_prev)  
        return xt_prev

    @torch.inference_mode()
    def i3sb_sampling(self, steps, pred_x0_fn, x1, ita, mask=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)
        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='I3SB sampling', total=len(steps)-1) if verbose else pair_steps
        cnt = 0
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            #print(ot_ode)
            if cnt<1:
                xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            else:
                xt = self.i3sb_p_posterior(prev_step, step, xt, x1, pred_x0, ita)        
            if mask is not None:
                
                # import matplotlib.pyplot as plt
                
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                # if cnt % 10 == 0:
                #     plt.subplot(1,2,1)
                #     plt.imshow(clear_color(xt))
                xt = (1. - mask) * xt_true + mask * xt
                #     plt.subplot(1,2,2)
                #     plt.imshow(clear_color(xt))
                #     plt.show()
                    
                #     import ipdb; ipdb.set_trace()

            cnt += 1
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
