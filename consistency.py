# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

from typing import List, Optional

import torch

from dnr_dataset import SOURCE_NAMES


def mixture_consistency(
    mixture: torch.Tensor,
    estimated_sources: torch.Tensor,
    source_dim: int,
    source_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Postprocessing for adding residual between mixture and estimated sources back to estimated sources.

    :param mixture (torch.tensor): audio mixture signal
    :param estimated_sources (torch.tensor): estimated separated source signals with added source dimension
    :param source_dim (int): dimension of sources in estimated source tensor
    :param source_weights (list): the weights for each source. Length must match the source_dim of estimated sources
    :return:
    """
    if source_weights is None:
        n_sources = estimated_sources.shape[source_dim]
        source_weights = [1 / n_sources] * n_sources
    source_weights = torch.tensor(source_weights).to(estimated_sources)
    source_weights = source_weights / source_weights.sum()
    n_trailing_dims = len(estimated_sources.shape[source_dim + 1 :])
    source_weights = source_weights.reshape(source_weights.shape + (1,) * n_trailing_dims)
    res = mixture - estimated_sources.sum(source_dim)
    new_source_signals = estimated_sources + source_weights * res.unsqueeze(source_dim)
    return new_source_signals


def dnr_consistency(mixture: torch.Tensor, estimated_sources: torch.Tensor, mode: str = "pass") -> torch.Tensor:
    """
    Postprocessing for adding residual between mixture and estimated sources back to estimated sources.

    :param mixture (torch.tensor): 3D Tensor with shape [batch, channels, samples]
                                   or 2D Tensor of shape [channels, samples]
    :param estimated_sources (torch.tensor): 4D Tensor with shape [batch, num_sources, channels, samples]
                                             or 3D Tensor of shape [num_sources, channels, samples]
    :param mode (str): choices=["all", "pass", "music_sfx"],
                       Whether to add the residual to estimates, 'pass' doesn't add residual, 'all' splits residual
                       among all sources, 'music_sfx' splits residual among music and sfx sources . (default: pass)"
    :return: Tensor of same shape as estimated sources
    """
    input_ndim = mixture.ndim
    if input_ndim > 2:  # we have a batch dimension
        source_dim = 1
    else:
        source_dim = 0
    if mode == "all":
        return mixture_consistency(mixture, estimated_sources, source_dim)
    elif mode == "music_sfx":
        source_weights = [0 if src == "speech" else 0.5 for src in SOURCE_NAMES]
        return mixture_consistency(mixture, estimated_sources, source_dim, source_weights)
    else:
        return estimated_sources
