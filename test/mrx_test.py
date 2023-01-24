# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

import torch

import mrx


def test_mrx_output_shape_with_batch():
    n_sources = 4
    n_channels = 2
    n_samples = 20000
    n_batch = 5
    input = torch.rand((n_batch, n_channels, n_samples))
    model = mrx.MRX(n_sources=n_sources)
    output = model(input)
    assert output.shape == (n_batch, n_sources, n_channels, n_samples)


def test_mrx_output_shape_no_batch():
    n_sources = 4
    n_channels = 2
    n_samples = 20000
    input = torch.rand((n_channels, n_samples))
    model = mrx.MRX(n_sources=n_sources)
    output = model(input)
    assert output.shape == (n_sources, n_channels, n_samples)
