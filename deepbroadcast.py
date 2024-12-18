# =============================================================================
# THIS repo is contributed by Mingze Gong (Graduated Student Member, IEEE) and 
# Shuoyao Wang (Senior Member, IEEE), Shenzhen University, Shenzhen, China. 
# This repo aims at introducing the DeepBroadcast method, with the 2-user case 
# study. The following is the network implementation part. 
# =============================================================================
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import utils_cm

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,bias=False)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv=conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn=nn.BatchNorm2d(out_channels)
        self.prelu=nn.PReLU()
    def forward(self, x): 
        out=self.conv(x)
        out=self.bn(out)
        out=self.prelu(out)
        return out

class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_block, self).__init__()
        self.deconv=deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  output_padding=output_padding)
        self.bn=nn.BatchNorm2d(out_channels)
        self.prelu=nn.PReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self, x, activate_func='prelu'): 
        out=self.deconv(x)
        out=self.bn(out)
        if activate_func=='prelu':
            out=self.prelu(out)
        elif activate_func=='sigmoid':
            out=self.sigmoid(out)
        return out   

class ChannelAwareNonLocalBlock(nn.Module):
    def __init__(self, channel, image_size, num_device=1):
        super().__init__()
        self.inter_channel = channel // 2
        self.H = image_size[0]
        self.W = image_size[1]
        self.csi_dim = int(self.inter_channel * self.H * self.W)
        self.num_device = num_device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  
        self.csi_theta = nn.Linear(self.num_device, self.csi_dim)
        
        
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        # self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, snr):
        # [N, C, H , W]
        b, c, h, w = x.size()
        if self.num_device == 1:
            snr = torch.tensor([snr]).to(self.device)
        elif self.num_device > 1:
            snr = torch.Tensor(snr).to(self.device)
        csi_theta = self.csi_theta(snr).unsqueeze(0).expand(b, -1).view(b, -1, c//2).contiguous()        
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c//2, -1)
        # [N, H * W, C/2]
        # x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        # mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = torch.matmul(csi_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class broadcastAttention (nn.Module):
    def __init__ (self, dim, num_device=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_device = num_device
        self.query_csi = nn.Sequential(nn.Linear(self.num_device, int(self.dim)),
                                       nn.ReLU(),
                                       nn.Linear(self.dim, int(self.dim)),
                                       nn.ReLU(),
                                       nn.Linear(self.dim, int(self.dim)),
                                       nn.Sigmoid()
                                       )
        self.key = nn.Sequential(nn.Linear(self.dim, self.dim),
                                 nn.Sigmoid())
        self.value = nn.Linear(self.dim, self.dim)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(self.dim, self.dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.layer_norm = nn.LayerNorm(self.dim)
                
    def forward (self, x, snr):
        # x: tensor, [B, H*W//P^2, dim]=[B, C, D](num_device=1) / [B, C, D*num_device](num_device>1)
        # snr: float(num_device=1) / list(num_device>1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        B, C = x.size()
        if self.num_device == 1:
            snr = torch.tensor([snr]).to(device)
        elif self.num_device > 1:
            snr = torch.Tensor(snr).to(device)
        
        x_norm = self.layer_norm(x)
        k = self.key(x_norm)
        v = self.value(x_norm)
        kv = k*v
        q = self.query_csi(snr.to(device)).unsqueeze(0).expand(B, C)    
        attn = x + q*kv    # attn: [B, C, C]
        attn = self.attn_drop(attn)
        res = self.proj(attn)
        
        return res
    
class DeepBroadcast (nn.Module):
    def __init__(self, n_c=20):
        super().__init__()
        self.num_lca = 3
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.1)
        # Encoder
        self.conv1 = conv_block(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(8, 8, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv_block(8, 8, kernel_size=3, stride=2, padding=1)
        self.conv4 = conv_block(8, 8, kernel_size=3, stride=1, padding=1)
        
        # Towards Rx1
        self.lca1 = nn.ModuleList()
        for i in range(self.num_lca):
            self.lca1.append(ChannelAwareNonLocalBlock(channel=8, image_size=(4, 4), num_device=1))
        self.cps_conv_rx1 = conv_block(8, 4, kernel_size=3, stride=1, padding=1)
        self.IB_mu_rx1 = nn.Sequential(nn.Linear(64, 32),
                                       nn.Tanh())
        self.IB_sigma_rx1 = nn.Sequential(nn.Linear(64, 32),
                                          nn.Sigmoid())
        self.gcf_1 = broadcastAttention(dim=32, num_device=2)
        # Towards Rx2
        self.lca2 = nn.ModuleList()
        for i in range(self.num_lca):
            self.lca2.append(ChannelAwareNonLocalBlock(channel=8, image_size=(4, 4), num_device=1))
        self.cps_conv_rx2 = conv_block(8, 4, kernel_size=3, stride=1, padding=1)
        self.IB_mu_rx2 = nn.Sequential(nn.Linear(64, 32),
                                       nn.Tanh())
        self.IB_sigma_rx2 = nn.Sequential(nn.Linear(64, 32),
                                          nn.Sigmoid())
        self.gcf_2 = broadcastAttention(dim=32, num_device=2)
        # Channel Encoder
        self.fusion = nn.Sequential(nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 16))
        
        ## Rx1
        self.rx1_fc = nn.Sequential(nn.Linear(16, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 2),
                                    nn.Softmax(dim=1))
        ## Rx2
        self.rx2_fc = nn.Sequential(nn.Linear(16, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 2),
                                    nn.Softmax(dim=1))
        
    def transmission(self, signal, snr, channel_flag):
        device = signal.device
        std = utils_cm.SNR2std(snr)
        signal = utils_cm.PowerNormalize(signal)
        if channel_flag == "AWGN":
            res = utils_cm.AWGN_channel(signal, std, device)
        elif channel_flag == "Rayleigh":
            res = utils_cm.Rayleigh_channel(signal, std, device)
        elif channel_flag == "Rician":
            res = utils_cm.Rician_channel(signal, std, device, K=2)
        return res
    
    def IB_cal(self, mu, sigma):
        B = mu.shape[0]
        mu = mu.view(B, -1)
        sigma = sigma.view(B, -1)
        KL_loss = utils_cm.KL_loss_function(mu, sigma)
        if self.training:
            eps = torch.randn_like(mu)
            x = (mu + torch.mul(eps,sigma))
        else:
            x = mu
        return x, KL_loss
    
    def task_channel_aware_sub_encoder(self, x, snr, rx):
        if rx == 1:
            for i in range(self.num_lca):
                x = self.lca1[i](x, float(snr['Rx1']))
            x = self.cps_conv_rx1(x).view(x.shape[0], -1)
            mu = 10 * self.IB_mu_rx1 (x)
            sigma = self.IB_sigma_rx1(x)
            x, KL_loss = self.IB_cal(mu, sigma)
        elif rx == 2:
            for i in range(self.num_lca):
                x = self.lca2[i](x, float(snr['Rx2']))
            x = self.cps_conv_rx2(x).view(x.shape[0], -1)
            mu = 10 * self.IB_mu_rx2 (x)
            sigma = self.IB_sigma_rx2(x)
            x, KL_loss = self.IB_cal(mu, sigma)
        return x, KL_loss
    
    def channel_aware_feature_fusion(self, x1, x2, snr_dict):
        snr = [snr_dict['Rx1'], snr_dict['Rx2']]
        x1 = self.gcf_1(x1, snr)
        x2 = self.gcf_2(x2, snr)
        x_cat = torch.cat((x1, x2), dim=1)
        bcs = self.fusion(x_cat)
        return bcs
    
    def Encoding(self, x, snr):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Task-channel aware sub encoding
        x1, KL_loss1 = self.task_channel_aware_sub_encoder(x, snr, rx=1)
        x2, KL_loss2 = self.task_channel_aware_sub_encoder(x, snr, rx=2)
        KL_loss = (KL_loss1+KL_loss2) / 2
        # Channel-aware feature fusion
        bcs = self.channel_aware_feature_fusion(x1, x2, snr)
        return bcs, KL_loss
    
    def forward(self, x, snr_dict:dict, channel_flag:dict):
        B = x.shape[0]
        # Encoding
        bcs, KL_loss = self.Encoding(x, snr_dict)
               
        # Transmission
        # Rx1
        rx1 = self.transmission(bcs, snr_dict['Rx1'], channel_flag['Rx1'])
        res1 = self.rx1_fc(rx1)
        # Rx2
        rx2 = self.transmission(bcs, snr_dict['Rx2'], channel_flag['Rx2'])
        res2 = self.rx2_fc(rx2)
        
        return res1, res2, KL_loss



    
