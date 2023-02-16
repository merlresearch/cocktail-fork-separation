# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

import torch

from consistency import mixture_consistency


def test_mixture_consistency_no_weights():
    n_sources = 4
    n_channels = 2
    n_samples = 20000
    n_batch = 5
    source_dim = 1
    true_sources = torch.rand((n_batch, n_sources, n_channels, n_samples))
    mixture = true_sources.sum(dim=source_dim)
    res = torch.rand(mixture.shape)
    mixture += res
    new_est = mixture_consistency(mixture, true_sources, source_dim)
    # if no source weights are given add the residual equally to all sources
    expected_output = true_sources + (1 / n_sources) * res.unsqueeze(source_dim)
    torch.testing.assert_close(new_est, expected_output)


def test_mixture_consistency_source_weights():
    n_sources = 4
    n_channels = 2
    n_samples = 20000
    n_batch = 5
    source_dim = 1
    source_weights = [0.1, 0, 0.9, 0]
    true_sources = torch.rand((n_batch, n_sources, n_channels, n_samples))
    mixture = true_sources.sum(dim=source_dim)
    res = torch.rand(mixture.shape)
    mixture += res
    new_est = mixture_consistency(mixture, true_sources, source_dim, source_weights)
    for i_src in range(n_sources):
        expected_output = true_sources[:, i_src, :, :] + source_weights[i_src] * res
        torch.testing.assert_close(new_est[:, i_src, :, :], expected_output)


def test_mixture_consistency_no_residual_unchanged():
    n_sources = 4
    n_channels = 2
    n_samples = 20000
    n_batch = 5
    source_dim = 1
    true_sources = torch.rand((n_batch, n_sources, n_channels, n_samples))
    mixture = true_sources.sum(dim=source_dim)
    new_est = mixture_consistency(mixture, true_sources, source_dim)
    torch.testing.assert_close(new_est, true_sources)
