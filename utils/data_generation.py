import pandas as pd
import numpy as np
import torch

def generate_time_varying_ggm(n, nt, d, threshold, seed):
    torch.manual_seed(seed)
    G = torch.rand(d,d) + torch.eye(d) < threshold
    G = G + G.T
    
    Xqt = torch.zeros(n, d, nt)
    t = torch.rand(nt)
    base = torch.randn(d, 2*d)
    base = base @ base.T/(2*d)

    for i in range(nt):
        mu = torch.zeros(1,d)
        Theta = torch.eye(d)
        Theta = Theta + base
        Theta.fill_diagonal_(2.0)
        Theta[G == 1] = 0.5*torch.sin(t[i]*10)

        Cov = torch.inverse(Theta)
        Xqt[:, :, i] = torch.distributions.MultivariateNormal(mu, Cov).sample((n,)).squeeze()
    
    return Xqt, t