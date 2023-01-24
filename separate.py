# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from dnr_dataset import EXT, SAMPLE_RATE, SOURCE_NAMES
from mrx import MRX

DEFAULT_PRE_TRAINED_MODEL_PATH = Path("checkpoints") / "default_mrx_pre_trained_weights.pth"


def load_default_pre_trained():
    model = MRX().eval()
    state_dict = torch.load(DEFAULT_PRE_TRAINED_MODEL_PATH)
    model.load_state_dict(state_dict)
    return model


def _mrx_output_to_dict(output):
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


def separate_soundtrack(audio_tensor, separation_model=None, device=None):
    """
    Separates a torch.Tensor into three stems. If a separation_model is provided, it will be used,
    otherwise the included pre-trained weights will be used.

    :param audio_tensor (torch.tensor): 2D Tensor of shape [channels, samples]
    :param separation_model (MRX, optional): a preloaded MRX model, or none to use included
                                             pre-trained model.
    :param device (int, optional): The gpu device for model inference. (default: -1) [cpu]
    :return: (dictionary): {'music': music_samples, 'speech': speech_samples, 'sfx': sfx_samples}
                            where each of the x_samples are 2D Tensor of shape [channels, samples]
    """
    if separation_model is None:
        separation_model = load_default_pre_trained()
    if device is not None:
        separation_model = separation_model.to(device)
        audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        output_tensor = separation_model(audio_tensor)
    return _mrx_output_to_dict(output_tensor)


def separate_soundtrack_file(audio_filepath, output_directory, separation_model=None, device=None):
    """
    Takes the path to a wav file, separates it, and saves the results in speech.wav, music.wav, and sfx.wav.
    Wraps seperate_soundtrack(). Audio will be resampled if it's not at the correct samplerate.

    :param audio_filepath (Path): path to mixture audio file to be separated
    :param output_directory (Path): directory where separated audio files will be saved
    :param separation_model (MRX, optional): a preloaded MRX model, or none to use included
                                             pre-trained model.
    :param device (int, optional): The gpu device for model inference. (default: -1) [cpu]
    """
    audio_tensor, fs = torchaudio.load(audio_filepath)
    if fs != SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(audio_tensor, fs, SAMPLE_RATE)
    output_dict = separate_soundtrack(audio_tensor, separation_model, device)
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
    args = parser.parse_args()
    if args.gpu_device != -1:
        device = torch.device("cuda:" + str(args.gpu_device))
    else:
        device = torch.device("cpu")
    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    separate_soundtrack_file(args.audio_path, output_dir, device=device)


if __name__ == "__main__":
    cli_main()
