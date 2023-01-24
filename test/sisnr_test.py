# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

import torch

from si_snr import si_snr


def test_si_snr_scale_invariant():
    n_batch, n_srcs, n_channels, n_samples = 5, 3, 1, 2000
    targets = torch.rand((n_batch, n_srcs, n_channels, n_samples))
    estimation_errors = 0.1 * torch.rand(targets.shape)
    estimates = targets + estimation_errors
    estimates_scale = 0.5 * estimates
    loss_no_scale = si_snr(estimates, targets)
    loss_scale = si_snr(estimates_scale, targets)
    torch.testing.assert_allclose(loss_scale, loss_no_scale)
