import numpy as np

import torch
import torch.nn as nn


class PointFlowNLL(nn.Module):
    def __init__(self):
        super(PointFlowNLL, self).__init__()

    def forward(self, samples, mus, logvars):
        return 0.5 * torch.add(
            torch.sum(sum(logvars) + ((samples[0] - mus[0])**2 / torch.exp(logvars[0]))) / samples[0].shape[0],
            np.log(2.0 * np.pi) * samples[0].shape[1] * samples[0].shape[2]
        )

class GaussianEntropy(nn.Module):
    def __init__(self):
        super(GaussianEntropy, self).__init__()

    def forward(self, logvars):
        return 0.5 * torch.add(logvars.shape[1] * (1.0 + np.log(2.0 * np.pi)), logvars.sum(1).mean())

class GaussianFlowNLL(nn.Module):
    def __init__(self):
        super(GaussianFlowNLL, self).__init__()

    def forward(self, samples, mus, logvars):
        return 0.5 * torch.add(
            torch.sum(sum(logvars) + ((samples[0] - mus[0])**2 / torch.exp(logvars[0]))) / samples[0].shape[0],
            np.log(2.0 * np.pi) * samples[0].shape[1]
        )

class Conditional_RNVP_loss(nn.Module):
    def __init__(self, **kwargs):
        super(Conditional_RNVP_loss, self).__init__()
        self.pnll_weight = kwargs.get('pnll_weight')
        self.gnll_weight = kwargs.get('gnll_weight')
        self.gent_weight = kwargs.get('gent_weight')
        self.PNLL = PointFlowNLL()
        self.GNLL = GaussianFlowNLL()               #may need to change to log_prob(z)
        self.GENT = GaussianEntropy()

    def forward(self, input, outputs):
        pnll = self.PNLL(outputs['p_prior_samples'], outputs['p_prior_mus'], outputs['p_prior_logvars'])
        gent = self.GENT(outputs['g_posterior_logvars'])
        gnll = self.GNLL(outputs['g_prior_samples'], outputs['g_prior_mus'], outputs['g_prior_logvars'])
        #return self.pnll_weight * pnll + self.gnll_weight * gnll - self.gent_weight * gent, pnll, gnll, gent
        return self.pnll_weight * pnll + self.gnll_weight * gnll, pnll, gnll

class Conditional_RNVP_with_image_prior_loss(nn.Module):
    def __init__(self, **kwargs):
        super(Conditional_RNVP_with_image_prior_loss, self).__init__()
        self.pnll_weight = kwargs.get('pnll_weight')
        self.PNLL = PointFlowNLL()

    def forward(self, input, outputs):
        pnll = self.PNLL(outputs['p_prior_samples'], outputs['p_prior_mus'], outputs['p_prior_logvars'])
        return self.pnll_weight * pnll, pnll

