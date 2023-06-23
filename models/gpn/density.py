from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
import math
import pyblaze.nn.functional as X

from models.gpn.normalizing_flow import NormalizingFlow, BatchedNormalizingFlowDensity
from models.gpn.mixture_density import MixtureDensity


class Evidence(nn.Module):
    """layer to transform density values into evidence representations according to a predefined scale"""

    def __init__(self,
                 scale: str,
                 tau: Optional[float] = None):
        super().__init__()
        self.tau = tau

        assert scale in ('latent-old', 'latent-new', 'latent-new-plus-classes', None)
        self.scale = scale

    def forward(self, log_q_c: Tensor, dim: int, **kwargs) -> Tensor:
        scaled_log_q = log_q_c + self.log_scale(dim, **kwargs)

        if self.tau is not None:
            scaled_log_q = self.tau * (scaled_log_q / self.tau).tanh()

        scaled_log_q = scaled_log_q.clamp(min=-30.0, max=30.0)

        return scaled_log_q

    def log_scale(self, dim: int, further_scale: int = 1) -> float:
        scale = 0

        if 'latent-old' in self.scale:
            scale = 0.5 * (dim * math.log(2 * math.pi) + math.log(dim + 1))
        if 'latent-new' in self.scale:
            scale = 0.5 * dim * math.log(4 * math.pi)

        scale = scale + math.log(further_scale)

        return scale


class Density(nn.Module):
    """
    encapsulates the PostNet step of transforming latent space
    embeddings z into alpha-scores with normalizing flows
    """

    def __init__(self,
                 dim_latent: int,
                 num_mixture_elements: int,
                 radial_layers: int = 16,
                 maf_layers: int = 0,
                 gaussian_layers: int = 0,
                 flow_size: float = 0.5,
                 maf_n_hidden: int = 0,
                 flow_batch_norm: bool = True,
                 use_batched_flow: bool = True):

        super().__init__()
        self.num_mixture_elements = num_mixture_elements
        self.dim_latent = dim_latent
        self.use_batched_flow = use_batched_flow

        self.use_flow = True
        if (maf_layers == 0) and (radial_layers == 0):
            self.use_flow = False

        if self.use_batched_flow:
            self.use_flow = False

        if self.use_batched_flow:
            self.flow = BatchedNormalizingFlowDensity(
                c=num_mixture_elements,
                dim=dim_latent,
                flow_length=radial_layers,
                flow_type='radial_flow')

        elif self.use_flow:
            self.flow = nn.ModuleList([
                NormalizingFlow(
                    dim=self.dim_latent,
                    radial_layers=radial_layers,
                    maf_layers=maf_layers,
                    flow_size=flow_size,
                    n_hidden=maf_n_hidden,
                    batch_norm=flow_batch_norm)
                for _ in range(num_mixture_elements)])

        else:
            self.flow = nn.ModuleList([MixtureDensity(
                dim=self.dim_latent,
                n_components=gaussian_layers) for _ in range(num_mixture_elements)])

    def forward(self, z: Tensor) -> Tensor:
        # produces log p(z|c)
        if self.use_batched_flow:
            log_q_c = self.forward_batched(z)

        elif self.use_flow:
            log_q_c = self.forward_flow(z)

        else:
            log_q_c = self.forward_mixture(z)

        if not self.training:
            # If we're evaluating and observe a NaN value, this is always caused by the
            # normalizing flow "diverging". We force these values to minus infinity.
            log_q_c[torch.isnan(log_q_c)] = float('-inf')

        return log_q_c

    def forward_batched(self, z: Tensor) -> Tensor:
        return self.flow.log_prob(z).transpose(0, 1)

    def forward_flow(self, z: Tensor) -> Tensor:
        n_nodes = z.size(0)
        log_q = torch.zeros((n_nodes, self.num_mixture_elements)).to(z.device.type)

        for c in range(self.num_mixture_elements):
            out, log_det = self.flow[c](z)
            log_p = X.log_prob_standard_normal(out) + log_det
            log_q[:, c] = log_p

        return log_q

    def forward_mixture(self, z: Tensor) -> Tensor:
        n_nodes = z.size(0)
        log_q = torch.zeros((n_nodes, self.num_mixture_elements)).to(z.device.type)

        for c in range(self.num_mixture_elements):
            log_q[:, c] = self.flow[c](z)

        return log_q