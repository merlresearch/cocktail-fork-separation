# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union

import pyloudnorm
import torch
import torchaudio

from consistency import dnr_consistency
from dnr_dataset import EXT, SAMPLE_RATE, SOURCE_NAMES
from mrx import MRX

DEFAULT_PRE_TRAINED_MODEL_PATH = Path("checkpoints") / "default_mrx_pre_trained_weights.pth"


def load_pre_trained(path=DEFAULT_PRE_TRAINED_MODEL_PATH):
    model = MRX().eval()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model


def _mrx_output_to_dict(output: torch.tensor) -> dict:
    """
    Convert MRX() to dictionary with one key per output source.

    :param output (torch.tensor): 3D Tensor of shape [3, channels, samples]
    :return: (dictionary): {'music': music_samples, 'speech': speech_samples, 'sfx': sfx_samples}
                            where each of the x_samples are 2D Tensor of shape [channels, samples]
    """
    output_dict = {}
    for src_name, audio_data in zip(SOURCE_NAMES, output):
        output_dict[src_name] = audio_data
    return output_dict


def _compute_gain(audio_tensor: torch.tensor, target_lufs: float) -> float:
    """
    Compute the gain required to achieve a target integrated loudness.

    :param audio_tensor (torch.tensor): 2D Tensor of shape [channels, samples].
    :param target_lufs (float): Target level in loudness units full scale.
    :return gain (float): Gain that when multiplied by audio_tensor will achieve target_lufs
    """
    meter = pyloudnorm.Meter(SAMPLE_RATE)
    loudness = meter.integrated_loudness(audio_tensor.cpu().numpy().T)
    gain_lufs = target_lufs - loudness
    gain = 10 ** (gain_lufs / 20.0)
    return gain


def separate_soundtrack(
    audio_tensor: torch.tensor,
    separation_model: Optional[Union[MRX, str, Path]] = None,
    device: Optional[int] = None,
    consistency_mode: Optional[str] = "pass",
    input_lufs: Optional[float] = -27.0,
):
    """
    Separates a torch.Tensor into three stems. If a separation_model is provided, it will be used,
    otherwise the included pre-trained weights will be used.

    :param audio_tensor (torch.tensor): 2D Tensor of shape [channels, samples]. Assumed samplerate of 44.1 kHz.
    :param separation_model (MRX, optional): a preloaded MRX model, a path to a pre-trained MRX
                                             model, or none to use included pre-trained model.
    :param device (int, optional): The gpu device for model inference. (default: -1) [cpu]
    :param consistency_mode (str, optional): choices=["all", "pass", "music_sfx"],
                                             Whether to add the residual to estimates, 'pass' doesn't add residual,
                                             'all' splits residual among all sources, 'music_sfx' splits residual among
                                              only music and sfx sources . (default: pass)"
    :param input_lufs (float, optional): Add gain to input and normalize output, so audio input level matches average
                                         of Divide and Remaster dataset in loudness units full scale.
                                         Pass None to skip. (default: -27)
    :return: (dictionary): {'music': music_samples, 'speech': speech_samples, 'sfx': sfx_samples}
                            where each of the x_samples are 2D Tensor of shape [channels, samples]
    """
    if separation_model is None:
        separation_model = load_pre_trained()
    elif isinstance(separation_model, (str, Path)):
        separation_model = load_pre_trained(separation_model)
    if device is not None:
        separation_model = separation_model.to(device)
        audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        if input_lufs is not None:
            gain = _compute_gain(audio_tensor, input_lufs)
            audio_tensor *= gain
        output_tensor = separation_model(audio_tensor)
        output_tensor = dnr_consistency(audio_tensor, output_tensor, mode=consistency_mode)
        if input_lufs is not None:
            output_tensor /= gain
    return _mrx_output_to_dict(output_tensor)


def separate_soundtrack_file(
    audio_filepath: Union[str, Path],
    output_directory: Union[str, Path],
    separation_model: Optional[Union[MRX, str, Path]] = None,
    device: Optional[int] = None,
    consistency_mode: Optional[str] = "pass",
    input_lufs: Optional[float] = -27.0,
) -> None:
    """
    Takes the path to a wav file, separates it, and saves the results in speech.wav, music.wav, and sfx.wav.
    Wraps seperate_soundtrack(). Audio will be resampled if it's not at the correct samplerate.

    :param audio_filepath (Path): path to mixture audio file to be separated
    :param output_directory (Path): directory where separated audio files will be saved
    :param separation_model (MRX, Path, optional): a preloaded MRX model, a path to a pre-trained MRX
                                             model, or none to use included pre-trained model.
    :param device (int, optional): The gpu device for model inference. (default: -1) [cpu]
    :param consistency_mode (str, optional): choices=["all", "pass", "music_sfx"],
                                             Whether to add the residual to estimates, 'pass' doesn't add residual,
                                             'all' splits residual among all sources, 'music_sfx' splits residual among
                                              only music and sfx sources . (default: pass)"
    :param input_lufs (float, optional): Add gain to input and normalize output, so audio input level matches average
                                         of Divide and Remaster dataset in loudness units full scale. (default: -27)
    """
    audio_tensor, fs = torchaudio.load(audio_filepath)
    if fs != SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(audio_tensor, fs, SAMPLE_RATE)
    output_dict = separate_soundtrack(
        audio_tensor, separation_model, device, consistency_mode=consistency_mode, input_lufs=input_lufs
    )
    for k, v in output_dict.items():
        output_path = Path(output_directory) / f"{k}{EXT}"
        torchaudio.save(output_path, v.cpu(), SAMPLE_RATE)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--audio-path",
        type=Path,
        help="Path to audio file to be separated in speech, music and, sound effects stems.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./separated_output"),
        help="Path to directory for saving output files.",
    )
    parser.add_argument("--gpu-device", default=-1, type=int, help="The gpu device for model inference. (default: -1)")
    parser.add_argument(
        "--mixture-residual",
        default="pass",
        type=str,
        choices=["all", "pass", "music_sfx"],
        help="Whether to add the residual to estimates, 'pass' doesn't add residual, 'all' splits residual among "
        "all sources, 'music_sfx' splits residual among only music and sfx sources . (default: pass)",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        type=str,
        help="Path to checkpoint file. If not provided, the included pre-trained weights will be used.",
    )
    args = parser.parse_args()
    if args.gpu_device != -1:
        device = torch.device("cuda:" + str(args.gpu_device))
    else:
        device = torch.device("cpu")
    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    separate_soundtrack_file(args.audio_path, output_dir, separation_model=args.ckpt_path,
                             device=device, consistency_mode=args.mixture_residual)


if __name__ == "__main__":
    cli_main()
