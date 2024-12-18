import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
import random
import os 
import torch.nn.functional as F
# =============================================================================
# Referring the implementation of ``Deep learning enabled semantic communication 
# systems'', whose authors are Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and 
# Biing-Hwang Juang, and the Github link is https://github.com/13274086/DeepSC
# =============================================================================
def SNR2std(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x

def AWGN_channel (x, std, device):
    return x+torch.normal(mean=0, std=std, size=x.shape).to(device)

def Rayleigh_channel (x, std, device):
    H_real = torch.normal(mean=0, std=math.sqrt(1/2), size=(1,1)).to(device)
    H_imag = torch.normal(mean=0, std=math.sqrt(1/2), size=(1,1)).to(device)
    H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
    x_hat = torch.matmul(x.view(x.shape[0], -1, 2), H)
    x_hat = AWGN_channel(x_hat, std, device)
    H = H.detach().cpu().numpy()
    H_inv = torch.from_numpy(np.linalg.inv(H)).to(x.device)
    x_hat = torch.matmul(x_hat, H_inv).view(x.shape).to(x.device)
    return x_hat

def Rician_channel (x, std, device, K=1):
    mean_normal = math.sqrt(K / (K+1))
    std_normal = math.sqrt(1 / (K+1))
    H_real = torch.normal(mean=mean_normal, std=std_normal, size=(1,1))
    H_imag = torch.normal(mean=mean_normal, std=std_normal, size=(1,1))
    H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
    x_hat = torch.matmul(x.view(x.shape[0], -1, 2), H)
    x_hat = AWGN_channel(x_hat, std, device)
    x_hat = torch.matmul(x_hat, torch.inverse(H)).view(x.shape)
    return x_hat

# =============================================================================
# Referring the implementation of ``Task-Oriented Communication for Multi-Device
# Cooperative Edge Inference'', whose authors are Jiawei Shao, Yuyi Mao, and Jun
# Zhang, and the Github link is https://github.com/shaojiawei07/VDDIB-SR/tree/main
# =============================================================================
def KL_loss_function(mu1,sigma1,sigma2 = 1):
    batch_size = mu1.size()[0]
    J = mu1.size()[1]

    mu_diff = (mu1) ** 2
    var1 = sigma1 ** 2
    var2 = sigma2 ** 2

    var_frac = var1 / var2
    diff_var_frac = mu_diff / var2

    term1 = torch.sum(torch.log(var_frac)) / batch_size
    term2 = torch.sum(var_frac) / batch_size
    term3 = torch.sum(diff_var_frac) / batch_size

    return - 0.5 * (term1 - term2 -term3 + J)




