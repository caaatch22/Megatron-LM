# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import os
from importlib.metadata import version
from typing import Callable

import torch
from pkg_resources import packaging
from torch import Tensor

from megatron.core.transformer.transformer_config import TransformerConfig


class LayerNorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        if config.normalization == 'LayerNorm':
            try:
                import apex

                from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

                HAVE_APEX = True
                isinstance = FusedLayerNorm(
                    config=config,
                    hidden_size=hidden_size,
                    eps=eps,
                    persist_layer_norm=config.persist_layer_norm,
                    zero_centered_gamma=config.layernorm_zero_centered_gamma,
                    normalization='LayerNorm'
                )
            except ImportError:
                import warnings
                from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm
                warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
                instance = WrappedTorchLayerNorm(
                    config=config,
                    hidden_size=hidden_size,
                    eps=eps,
                    persist_layer_norm=config.persist_layer_norm,
                    zero_centered_gamma=config.layernorm_zero_centered_gamma,
                    normalization='LayerNorm'
                )
        elif config.normalization == 'RMSNorm':
            try:
                import apex
                from megatron.core.fusions.fused_rms_norm import FusedRMSNorm
                HAVE_APEX = True
                instance = FusedRMSNorm(
                    config=config,
                    hidden_size=hidden_size,
                    eps=eps,
                    persist_layer_norm=config.persist_layer_norm,
                    zero_centered_gamma=config.layernorm_zero_centered_gamma,
                    normalization='RMSNorm'
                )
            except ImportError:
                import warnings
                from megatron.core.transformer.torch_layer_norm import WrappedTorchRMSNorm
                warnings.warn(f'Apex is not installed. Falling back to Torch RMSNorm')
                instance = WrappedTorchRMSNorm(
                    config=config,
                    hidden_size=hidden_size,
                    eps=eps,
                    persist_layer_norm=config.persist_layer_norm,
                    zero_centered_gamma=config.layernorm_zero_centered_gamma,
                    normalization='RMSNorm'
                )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance

