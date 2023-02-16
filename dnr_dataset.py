# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.data as data
import torchaudio

SOURCE_NAMES = ["music", "speech", "sfx"]
MIXTURE_NAME = "mix"
SAMPLE_RATE = 44100
EXT = ".wav"


class DivideAndRemaster(data.Dataset):
    def __init__(
        self,
        root_path: Union[str, Path],
        subset: str,
        chunk_size_sec: Optional[float] = None,
        random_start: Optional[bool] = False,
    ) -> None:
        """
        :param root_path: path to top level dataset directory
        :param subset: Options: [``"tr"``, ``"cv"``, ``"tt"``].
        :param chunk_size_sec: in seconds, instead of reading entire file, read only a chunk of this size
        :param random_start: If True and chunk_size_sec is specified, __get_item()___ will use a random start sample
        """
        self.path = os.path.join(root_path, subset)
        if not os.path.isdir(self.path):
            raise RuntimeError("Dataset not found. Please check root_path")
        if chunk_size_sec is not None:
            self.chunk_size = int(chunk_size_sec * SAMPLE_RATE)
        else:
            self.chunk_size = -1
        self.random_start = random_start
        self.track_list = self._get_tracklist()
        self._check_subset_lengths(subset)

    def _get_tracklist(self) -> List[str]:
        path = Path(self.path)
        names = []
        for root, folders, _ in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith(".") or folders or root == path:
                continue
            name = str(root.relative_to(path))
            names.append(name)
        return sorted(names)

    def _check_subset_lengths(self, subset: str):
        """
        Assert if the number of files is incorrect, to ensure we are using DnR v2 not an old version
        """
        if subset == "tr":
            assert len(self.track_list) == 3406, "Expected 3406 mix in training set"
        elif subset == "cv":
            assert len(self.track_list) == 487, "Expected 487 mix in validation set"
        elif subset == "tt":
            assert len(self.track_list) == 973, "Expected 973 mix in testing set"

    def _get_audio_path(self, track_name: str, source_name: str) -> Path:
        return Path(self.path) / track_name / f"{source_name}{EXT}"

    def _get_chunk_indices(self, track_name: str) -> Tuple[int, int]:
        mix_path = self._get_audio_path(track_name, MIXTURE_NAME)
        num_frames_total = torchaudio.info(mix_path).num_frames
        start_frame = 0
        num_frames_to_read = self.chunk_size
        if num_frames_total <= self.chunk_size:
            num_frames_to_read = -1
        else:
            if self.random_start:
                start_frame = int(torch.randint(0, num_frames_total - self.chunk_size, (1,)))
        return start_frame, num_frames_to_read

    def _read_audio(self, path: Path, frame_offset: Optional[int] = 0, num_frames: Optional[int] = -1) -> torch.Tensor:
        y, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        assert sr == SAMPLE_RATE, "audio samplerate of data does not match requested samplerate"
        return y

    def _load_track(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        track_name = self.track_list[index]
        frame_offset, num_frames = self._get_chunk_indices(track_name)
        mix_path = self._get_audio_path(track_name, MIXTURE_NAME)
        y_mix = self._read_audio(mix_path, frame_offset, num_frames)
        src_wavs = []
        for source_name in SOURCE_NAMES:
            src_path = self._get_audio_path(track_name, source_name)
            y_src = self._read_audio(src_path, frame_offset, num_frames)
            src_wavs.append(y_src)
        stacked_sources = torch.stack(src_wavs)
        return y_mix, stacked_sources, track_name

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load the index-th example from the dataset.

        :param index (int): sample index to load
        :return:
            Tuple of the following items;
            torch.Tensor:
                mixture [channels, samples]
            torch.Tensor:
                targets [sources, channels, samples]
            str:
                Dataset filename
        """
        return self._load_track(index)

    def __len__(self) -> int:
        return len(self.track_list)
