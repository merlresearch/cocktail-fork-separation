# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

from typing import Optional

import torch

EPSILON = 1e-8


def snr_loss(
    estimates: torch.Tensor, targets: torch.Tensor, dim: Optional[int] = -1, scale_invariant: Optional[bool] = True
) -> torch.Tensor:
    """
    Computes the negative [scale-invariant] signal (source) to noise (distortion) ratio.
    :param estimates (torch.Tensor): estimated source signals, tensor of shape [..., n_samples, ....]
    :param targets (torch.Tensor): ground truth signals, tensor of shape [...., n_samples, ....]
    :param dim (int): time (sample) dimension
    :param scale_invariant (bool): use SI-SNR when true, regular SNR when false
    :return (torch.Tensor): estimated [SI-]SNR with one value for each non-sample dimension
    """
    if scale_invariant:
        estimates = _mean_center(estimates, dim=dim)
        targets = _mean_center(targets, dim=dim)
        sig_power = _l2_square(targets, dim=dim, keepdim=True)  # [n_batch, 1, n_srcs]
        dot_ = torch.sum(estimates * targets, dim=dim, keepdim=True)
        scale = dot_ / (sig_power + 1e-12)
    else:  # Regular SNR with no processing
        scale = 1
    s_target = scale * targets
    e_noise = estimates - s_target
    si_snr_array = _l2_square(s_target, dim=dim) / (_l2_square(e_noise, dim=dim) + EPSILON)
    si_snr_array = -10 * torch.log10(si_snr_array + EPSILON)
    return si_snr_array


def _mean_center(arr: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    mn = torch.mean(arr, dim=dim, keepdim=True)
    return arr - mn


def _l2_square(arr: torch.Tensor, dim: Optional[int] = None, keepdim: Optional[bool] = False) -> torch.Tensor:
    return torch.sum(arr**2, dim=dim, keepdim=keepdim)
