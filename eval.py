# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dnr_dataset import DivideAndRemaster
from lightning_train import CocktailForkModule
from mrx import MRX
from separate import DEFAULT_PRE_TRAINED_MODEL_PATH


def _read_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt.keys():
        # lightning module checkpoint
        model = CocktailForkModule.load_from_checkpoint(checkpoint_path)
    else:
        # only network weights
        model = MRX()
        model.load_state_dict(ckpt)
        model = CocktailForkModule(model=model)
    return model


def _lightning_eval():
    parser = ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="The path to the DnR directory containing ``tr`` ``cv``  and ``tt`` directories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_PRE_TRAINED_MODEL_PATH,
        help="Path to trained model weights. Can be a pytorch_lightning checkpoint or pytorch state_dict",
    )
    parser.add_argument("--gpu-device", default=-1, type=int, help="The gpu device for model inference. (default: -1)")
    args = parser.parse_args()

    test_dataset = DivideAndRemaster(args.root_dir, "tt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        drop_last=False,
    )

    if args.gpu_device >= 0:
        devices = [args.gpu_device]
        accelerator = "gpu"
    else:
        devices = "auto"
        accelerator = "cpu"

    trainer = Trainer(
        devices=devices,
        accelerator=accelerator,
        enable_progress_bar=True,  # this will print the results to the command line
    )
    model = _read_checkpoint(args.checkpoint)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    _lightning_eval()
