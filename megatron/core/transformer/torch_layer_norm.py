# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings

import torch

from megatron.core.transformer import TransformerConfig


class WrappedTorchLayerNorm(torch.nn.LayerNorm):

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,  ## TODO: unused arguments. See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ):
        self.config = config
        assert (
            not self.config.layernorm_zero_centered_gamma
        ), f"zero_centered_gamma not supported by torch LayerNorm"

        assert (
            self.config.normalization == "LayerNorm"
        ), f'({self.config.normalization}) is not supported in by torch Layernorm'

        assert (
            not self.config.persist_layer_norm
        ), f"persist_layer_norm not supported by torch LayerNorm"

        assert (
            not self.config.sequence_parallel
        ), f"sequence parallel not supported by torch LayerNorm"

        assert (
            not self.config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        super().__init__(
            dim=hidden_size,  ## applied to last len(normalized_shape.size) dimensions
            eps=eps,
        )



class RMSNorm(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False):
        """RMS Normaliation module

        Args:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class WrappedTorchRMSNorm(RMSNorm):
    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",  # included to match TE interface
    ):
        self.config = config
        assert (
            not self.config.layernorm_zero_centered_gamma
        ), f"zero_centered_gamma not supported by torch RMSNorm"

        assert (
            self.config.normalization == "RMSNorm"
        ), f'({self.config.normalization}) is not supported in by torch RMSNorm'

        assert (
            not self.config.persist_layer_norm
        ), f"persist_layer_norm not supported by torch RMSNorm"

        assert (
            not self.config.sequence_parallel
        ), f"sequence parallel not supported by torch RMSNorm"

        assert (
            not self.config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch RMSNorm"

        super().__init__(
            dim=hidden_size,  ## applied to last len(normalized_shape.size) dimensions
            eps=eps,
        )
