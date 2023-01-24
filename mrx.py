# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

from typing import List, Optional

import torch


class MRX(torch.nn.Module):
    """MRX (Multi Resolution Cross Network) architecture from:
    *THE COCKTAIL FORK PROBLEM: THREE-STEM AUDIO SEPARATION FOR REAL-WORLD SOUNDTRACKS*

    Note:
     This implementation differs slightly from the version in the paper, in that we use
     a batchnorm layer on the input instead of global mean/variance standardization.

    Args:
        n_sources (int, optional): The number of sources to separate.
        window_lengths (List[int], optional): List of multiple STFT window sizes.
                                              Default is None, which uses [1024, 2048, 8192]
        n_hop (int, optional): STFT hop-size. Constant across resolutions.
        n_hidden (int, optional): The number of hidden units throughout the network.
        n_lstm_layers (int, optional): The number of LSTM layers.
    """

    def __init__(
        self,
        n_sources: Optional[int] = 3,
        window_lengths: Optional[List[int]] = None,
        n_hop: Optional[int] = 256,
        n_hidden: Optional[int] = 512,
        n_lstm_layers: Optional[int] = 3,
    ) -> None:
        super().__init__()
        if window_lengths is None:
            window_lengths = [1024, 2048, 8192]
        self._window_lengths = window_lengths
        self._n_hop = n_hop
        self._n_window_fft_bins = [wl // 2 + 1 for wl in window_lengths]
        self._encoders = torch.nn.ModuleList()
        for n_bins in self._n_window_fft_bins:
            self._encoders.append(_EncoderBlock(n_bins, n_hidden))
        self._cross_net = _CrossNet(n_sources, n_hidden, n_lstm_layers)
        self._decoders = torch.nn.ModuleList()
        for _ in range(n_sources):
            # For each source we have one decoder at each stft resolution
            src_dec_list = torch.nn.ModuleList()
            for n_bins in self._n_window_fft_bins:
                src_dec_list.append(_DecoderBlock(n_bins, n_hidden))
            self._decoders.append(src_dec_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform source separation. Generate audio source waveforms.
        Args:
            x (torch.Tensor): 3D Tensor with shape [batch, channels, samples]
                              or 2D Tensor of shape [channels, samples]
        Returns:
            Tensor: 4D Tensor with shape [batch, num_sources, channels, samples]
                    or 3D Tensor of shape [num_sources, channels, samples]
        """
        signal_length = x.shape[-1]
        input_ndim = x.ndim
        encoded_spectrograms = []
        original_spectrograms = []
        for window_length, enc_layer in zip(self._window_lengths, self._encoders):
            spec = _stft(x, window_length, self._n_hop)
            enc = enc_layer(spec)
            original_spectrograms.append(spec)
            encoded_spectrograms.append(enc)

        cross_net_output = self._cross_net(encoded_spectrograms)

        masked_source_signals = []
        for src_decoder in self._decoders:
            multi_res_signals = []
            for complex_spec, dec_layer in zip(original_spectrograms, src_decoder):
                mask = dec_layer(cross_net_output)
                mask = mask.transpose(-1, -2)  # swap time and frequency dimensions
                mask = mask.reshape(complex_spec.shape)
                masked_spec = mask * complex_spec
                multi_res_signals.append(_istft(masked_spec, self._n_hop, signal_length))
            masked_source = torch.sum(torch.stack(multi_res_signals), dim=0)
            masked_source_signals.append(masked_source)
        # Return the source signals stacked after batch dimension
        if input_ndim > 2:  # we have a batch dimension
            stack_dim = 1
        else:
            stack_dim = 0
        masked_source_signals = torch.stack(masked_source_signals, dim=stack_dim)
        return masked_source_signals


def _stft(signal: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> torch.Tensor:
    leading_dims = list(signal.shape[:-1])
    n_samples = int(signal.shape[-1])
    signal = signal.reshape(-1, n_samples)
    spectrogram = torch.stft(
        signal,
        n_fft,
        hop_length,
        window=torch.hann_window(n_fft).to(signal),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    _, n_freqs, n_frames = spectrogram.shape
    leading_dims.extend([n_freqs, n_frames])
    return spectrogram.view(leading_dims)


def _istft(spectrogram: torch.Tensor, hop_length: int, signal_length: int) -> torch.Tensor:
    leading_dims = list(spectrogram.shape[:-2])
    n_freqs = int(spectrogram.shape[-2])
    n_frames = int(spectrogram.shape[-1])

    n_fft = 2 * n_freqs - 2
    spectrogram = spectrogram.view(-1, n_freqs, n_frames)
    signal = torch.istft(
        spectrogram,
        n_fft,
        hop_length,
        window=torch.hann_window(n_fft).to(spectrogram.real),
        win_length=n_fft,
        normalized=True,
        length=signal_length,
        center=True,
    )
    _, length = signal.shape
    leading_dims.append(length)
    return signal.view(leading_dims)


class _EncoderBlock(torch.nn.Module):
    def __init__(self, num_inputs: int, num_hidden: int) -> None:
        """
        CrossNet encoder originally proposed in Sawata et al.
        "All for one and one for all: Improving music separation by bridging networks," ICASSP 2021.

        Our implementation differs in two main ways: (1) we use a BatchNorm layer on input instead of
        precomputing mean/std, and (2) we encode each channel independently to handle variable number
        of channels at inference time.

        :param num_inputs (int): STFT input dimension
        :param num_hidden (int): number of hidden layer units
        """
        super().__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_inputs),  # replaces mean-variance normalization from original XUMX
            torch.nn.Linear(num_inputs, num_hidden, bias=False),
            torch.nn.BatchNorm1d(num_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):  3D Tensor with shape [batch, channels, freqs, frames]
                               or 2D Tensor of shape [channels, freqs, frames]. Complex STFT.
        Returns:
            Tensor: 3D Tensor with shape [batch, channels, frames, num_hidden]
                    or 2D Tensor of shape [channels, frames, num_hidden]

        """
        n_freqs = x.shape[-2]
        n_frames = x.shape[-1]
        enc = x.transpose(-1, -2)  # swap time and frequency dimensions
        enc = enc.reshape(-1, n_freqs)
        enc = enc.abs()  # magnitude
        enc = self.layer(enc)
        enc = enc.reshape(-1, n_frames, enc.shape[-1])  # LSTM compatible shape
        return torch.tanh(enc)


class _DecoderBlock(torch.nn.Module):
    """
    CrossNet decode block originally proposed in Sawata et al.
    "All for one and one for all: Improving music separation by bridging networks," ICASSP 2021.
    The difference in our implementation is that each channel is decoded independently to handle variable number
    of channels at inference time.
    """

    def __init__(self, num_outputs: int, num_hidden: int) -> None:
        super().__init__()

        first_layer_num_inputs = 2 * num_hidden  # concatenated CrossNet LSTM inputs/outputs
        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_features=first_layer_num_inputs, out_features=num_hidden, bias=False),
            torch.nn.BatchNorm1d(num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_outputs, bias=False),
            torch.nn.BatchNorm1d(num_outputs),
            torch.nn.ReLU(),  # mask activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = x.shape[-2]
        n_feats = x.shape[-1]
        x = x.reshape(-1, n_feats)
        x = self.block(x)
        x = x.reshape(-1, n_frames, x.shape[-1])
        return x


class _CrossNet(torch.nn.Module):
    """
    Parallel LSTM branches with averaging layers. Architecture was originally proposed in Sawata et al.
    "All for one and one for all: Improving music separation by bridging networks," ICASSP 2021.
    In our multi-resolution extension each branch takes a different input.
    """

    def __init__(self, num_branches: int, num_hidden: int, num_layers: int) -> None:
        super().__init__()

        lstm_hidden_dim = num_hidden // 2  # bidirectional LSTM output will double size
        self.lstm = torch.nn.ModuleList()
        for _ in range(num_branches):
            self.lstm.append(
                torch.nn.LSTM(
                    input_size=num_hidden,
                    hidden_size=lstm_hidden_dim,
                    num_layers=num_layers,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.4,
                )
            )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        cross_1 = torch.mean(torch.stack(inputs), dim=0)
        cross_2 = torch.mean(torch.stack([layer(cross_1)[0] for layer in self.lstm]), dim=0)
        return torch.cat([cross_1, cross_2], dim=-1)  # use concat skip connection from original XUMX
