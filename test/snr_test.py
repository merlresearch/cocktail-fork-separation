# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

import torch

from snr import snr_loss


def test_si_snr_scale_invariant():
    n_batch, n_srcs, n_channels, n_samples = 5, 3, 1, 2000
    targets = torch.rand((n_batch, n_srcs, n_channels, n_samples))
    estimation_errors = 0.1 * torch.rand(targets.shape)
    estimates = targets + estimation_errors
    estimates_scale = 0.5 * estimates
    loss_no_scale = snr_loss(estimates, targets)
    loss_scale = snr_loss(estimates_scale, targets)
    torch.testing.assert_close(loss_scale, loss_no_scale)
