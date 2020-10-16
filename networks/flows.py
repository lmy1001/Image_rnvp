from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from networks.layers import SharedDot, Swish


class CondRealNVPFlow3D(nn.Module):
    def __init__(self, f_n_features, g_n_features,
                 weight_std=0.01, warp_inds=[0],
                 batch_norm=True,
                 centered_translation=False, eps=1e-6):
        super(CondRealNVPFlow3D, self).__init__()
        self.f_n_features = f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.warp_inds = warp_inds
        self.keep_inds = [0, 1, 2]
        self.centered_translation = centered_translation
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        self.batch_norm = batch_norm
        for ind in self.warp_inds:
            self.keep_inds.remove(ind)

        T_mu_0_layers = [('mu_sd0', SharedDot(len(self.keep_inds), self.f_n_features, 1))]
        if self.batch_norm:
            T_mu_0_layers.append(('mu_sd0_bn', nn.BatchNorm1d(self.f_n_features)))
        T_mu_0_layers.append(('mu_sd0_relu', nn.ReLU(inplace=True)))
        T_mu_0_layers.append(('mu_sd1', SharedDot(self.f_n_features, self.f_n_features, 1)))
        T_mu_0_layers.append(('mu_sd1_bn', nn.BatchNorm1d(self.f_n_features, affine=False)))
        self.T_mu_0 = nn.Sequential(OrderedDict([*T_mu_0_layers]))

        T_mu_0_cond_w_layers = [('mu_sd1_film_w0', nn.Linear(self.g_n_features, self.f_n_features, bias=False))]
        if self.batch_norm:
            T_mu_0_cond_w_layers.append(('mu_sd1_film_w0_bn', nn.BatchNorm1d(self.f_n_features)))
        T_mu_0_cond_w_layers.append(('mu_sd1_film_w0_swish', Swish()))
        T_mu_0_cond_w_layers.append(('mu_sd1_film_w1', nn.Linear(self.f_n_features, self.f_n_features, bias=True)))
        self.T_mu_0_cond_w = nn.Sequential(OrderedDict([*T_mu_0_cond_w_layers]))  # 512-64-64

        T_mu_0_cond_b_layers = [ ('mu_sd1_film_b0', nn.Linear(self.g_n_features, self.f_n_features, bias=False))]
        if self.batch_norm:
            T_mu_0_cond_b_layers.append(('mu_sd1_film_b0_bn', nn.BatchNorm1d(self.f_n_features)))
        T_mu_0_cond_b_layers.append(('mu_sd1_film_b0_swish', Swish()))
        T_mu_0_cond_b_layers.append(('mu_sd1_film_b1', nn.Linear(self.f_n_features, self.f_n_features, bias=True)))
        self.T_mu_0_cond_b = nn.Sequential(OrderedDict([*T_mu_0_cond_b_layers]))  # 512-64-64

        self.T_mu_1 = nn.Sequential(OrderedDict([
            ('mu_sd1_relu', nn.ReLU(inplace=True)),
            ('mu_sd2', SharedDot(self.f_n_features, len(self.warp_inds), 1, bias=True))
        ]))  # 128-1

        with torch.no_grad():
            self.T_mu_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_w[-1].bias.data, 0.0)
            self.T_mu_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_b[-1].bias.data, 0.0)
            self.T_mu_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_1[-1].bias.data, 0.0)

        T_logvar_0_layers = [('logvar_sd0', SharedDot(len(self.keep_inds), self.f_n_features, 1))]
        if self.batch_norm:
            T_logvar_0_layers.append(('logvar_sd0_bn', nn.BatchNorm1d(self.f_n_features)))
        T_logvar_0_layers.append(('logvar_sd0_relu', nn.ReLU(inplace=True)))
        T_logvar_0_layers.append(('logvar_sd1', SharedDot(self.f_n_features, self.f_n_features, 1)))
        if self.batch_norm:
            ('logvar_sd1_bn', nn.BatchNorm1d(self.f_n_features, affine=False))
        self.T_logvar_0 = nn.Sequential(OrderedDict([*T_logvar_0_layers]))     #3-64-64

        T_logvar_0_cond_w_layers = [('logvar_sd1_film_w0', nn.Linear(self.g_n_features, self.f_n_features, bias=False))]
        if self.batch_norm:
            T_logvar_0_cond_w_layers.append( ('logvar_sd1_film_w0_bn', nn.BatchNorm1d(self.f_n_features)))
        T_logvar_0_cond_w_layers.append(('logvar_sd1_film_w0_swish', Swish()))
        T_logvar_0_cond_w_layers.append(('logvar_sd1_film_w1', nn.Linear(self.f_n_features, self.f_n_features, bias=True)))
        self.T_logvar_0_cond_w = nn.Sequential(OrderedDict([*T_logvar_0_cond_w_layers]))     #512-64-64

        T_logvar_0_cond_b_layers = [('logvar_sd1_film_b0', nn.Linear(self.g_n_features, self.f_n_features, bias=False))]
        if self.batch_norm:
            T_logvar_0_cond_b_layers.append(('logvar_sd1_film_b0_bn', nn.BatchNorm1d(self.f_n_features)))
        T_logvar_0_cond_b_layers.append(('logvar_sd1_film_b0_swish', Swish()))
        T_logvar_0_cond_b_layers.append(('logvar_sd1_film_b1', nn.Linear(self.f_n_features, self.f_n_features, bias=True)))
        self.T_logvar_0_cond_b = nn.Sequential(OrderedDict([*T_logvar_0_cond_b_layers]))     #512-64-64

        self.T_logvar_1 = nn.Sequential(OrderedDict([
            ('logvar_sd1_relu', nn.ReLU(inplace=True)),
            ('logvar_sd2', SharedDot(self.f_n_features, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_logvar_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_w[-1].bias.data, 0.0)
            self.T_logvar_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_b[-1].bias.data, 0.0)
            self.T_logvar_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_1[-1].bias.data, 0.0)

    def forward(self, p, g, mode='direct'):
        logvar = torch.zeros_like(p)    # 10 * 3 * npoints
        mu = torch.zeros_like(p)

        logvar[:, self.warp_inds, :] = nn.functional.softsign(self.T_logvar_1(
            torch.add(self.eps, torch.exp(self.T_logvar_0_cond_w(g).unsqueeze(2))) *
            self.T_logvar_0(p[:, self.keep_inds, :].contiguous()) + self.T_logvar_0_cond_b(g).unsqueeze(2)
        ))      #只计算了 [10, 0, npoints]的值， 其他均为0

        mu[:, self.warp_inds, :] = self.T_mu_1(
            torch.add(self.eps, torch.exp(self.T_mu_0_cond_w(g).unsqueeze(2))) *
            self.T_mu_0(p[:, self.keep_inds, :].contiguous()) + self.T_mu_0_cond_b(g).unsqueeze(2)
        )

        logvar = logvar.contiguous()
        mu = mu.contiguous()

        if mode == 'direct':
            p_out = torch.sqrt(torch.add(self.eps, torch.exp(logvar))) * p + mu     # 每次只变一行， N*3*npoints(只有[:, 0, :]的位置值发生改
        elif mode == 'inverse':
            p_out = (p - mu) / torch.sqrt(torch.add(self.eps, torch.exp(logvar)))

        return p_out, mu, logvar


class CondRealNVPFlow3DTriple(nn.Module):
    def __init__(self, f_n_features, g_n_features, weight_std=0.02, pattern=0, centered_translation=False, batch_norm=True):
        super(CondRealNVPFlow3DTriple, self).__init__()
        self.f_n_features = f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.pattern = pattern
        self.centered_translation = centered_translation
        self.batch_norm = batch_norm

        if pattern == 0:
            self.nvp1 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0],
                                          batch_norm=self.batch_norm,
                                          centered_translation=centered_translation)
            self.nvp2 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[1],
                                          batch_norm=self.batch_norm,
                                          centered_translation=centered_translation)
            self.nvp3 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[2],
                                          batch_norm=self.batch_norm,
                                          centered_translation=centered_translation)
        elif pattern == 1:
            self.nvp1 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0, 1],
                                          batch_norm=self.batch_norm,
                                          centered_translation=centered_translation)
            self.nvp2 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0, 2],
                                          batch_norm=self.batch_norm,
                                          centered_translation=centered_translation)
            self.nvp3 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[1, 2],
                                          batch_norm=self.batch_norm,
                                          centered_translation=centered_translation)

    def forward(self, p, g, mode='direct'):
        if mode == 'direct':
            p1, mu1, logvar1 = self.nvp1(p, g, mode=mode)           #N* 3 * npoints, p1只有第一行pos[0]发生改变
            p2, mu2, logvar2 = self.nvp2(p1, g, mode=mode)          # 将变化过的p0带入，pos[1]发生变化
            p3, mu3, logvar3 = self.nvp3(p2, g, mode=mode)          #pos[2]发生变化
        elif mode == 'inverse':
            p3, mu3, logvar3 = self.nvp3(p, g, mode=mode)
            p2, mu2, logvar2 = self.nvp2(p3, g, mode=mode)
            p1, mu1, logvar1 = self.nvp1(p2, g, mode=mode)

        return [p1, p2, p3], [mu1, mu2, mu3], [logvar1, logvar2, logvar3]