# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from consistency import dnr_consistency
from dnr_dataset import SOURCE_NAMES, DivideAndRemaster
from mrx import MRX
from snr import snr_loss


class CocktailForkModule(LightningModule):
    def __init__(self, model=None, si_loss=True, mixture_residual="pass"):
        super().__init__()
        if model is None:
            self.model = MRX()
        else:
            self.model = model
        self.si_loss = si_loss
        self.mixture_residual = mixture_residual

    def _step(self, batch, batch_idx, split):
        x, y, filenames = batch
        y_hat = self.model(x)
        y_hat = dnr_consistency(x, y_hat, mode=self.mixture_residual)
        loss = snr_loss(y_hat, y, scale_invariant=self.si_loss).mean()
        self.log(f"{split}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        x, y, filenames = batch
        y_hat = self.model(x)
        y_hat = dnr_consistency(x, y_hat, mode=self.mixture_residual)
        est_sisdr = -snr_loss(y_hat, y, scale_invariant=True).mean(-1).mean(0)  # average of batch and channel
        est_snr = -snr_loss(y_hat, y, scale_invariant=False).mean(-1).mean(0)
        # expand mixture to shape of isolated sources for noisy SDR
        repeat_shape = len(y.shape) * [1]
        repeat_shape[1] = y.shape[1]
        x = x.unsqueeze(1).repeat(repeat_shape)
        noisy_sisdr = -snr_loss(x, y, scale_invariant=True).mean(-1).mean(0)
        noisy_snr = -snr_loss(x, y, scale_invariant=False).mean(-1).mean(0)
        result_dict = {}
        for i, src in enumerate(SOURCE_NAMES):
            result_dict[f"noisy_sisdr_{src}"] = noisy_sisdr[i].item()
            result_dict[f"est_sisdr_{src}"] = est_sisdr[i].item()
            result_dict[f"noisy_snr_{src}"] = noisy_snr[i].item()
            result_dict[f"est_snr_{src}"] = est_snr[i].item()
        self.log_dict(result_dict, on_epoch=True)
        return est_sisdr.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        lr_scheduler_config = {"scheduler": lr_scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}
        return [optimizer], [lr_scheduler_config]


def _get_dataloaders(
    dnr_root_dir: Union[str, Path],
    train_batch_size: int = 25,
    train_chunk_sec: float = 9.0,
    eval_batch_size: int = 5,
    num_workers: int = 4,
) -> Tuple[DataLoader]:

    train_dataset = DivideAndRemaster(dnr_root_dir, "tr", chunk_size_sec=train_chunk_sec, random_start=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    valid_dataset = DivideAndRemaster(dnr_root_dir, "cv")
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_dataset = DivideAndRemaster(dnr_root_dir, "tt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, valid_loader, test_loader


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--train-batch-size", default=20, type=int)
    parser.add_argument("--eval-batch-size", default=10, type=int)
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="The path to the DnR directory containing ``tr`` ``cv``  and ``tt`` directories.",
    )
    parser.add_argument(
        "--exp-dir", default=Path("./exp"), type=Path, help="The directory to save checkpoints and logs."
    )
    parser.add_argument(
        "--chunk-size",
        default=9.0,
        type=float,
        help="length of chunk from file for training in seconds. (default: 9.0)",
    )
    parser.add_argument(
        "--epochs",
        metavar="NUM_EPOCHS",
        default=200,
        type=int,
        help="The number of epochs to train. (default: 200)",
    )
    parser.add_argument(
        "--num-gpu",
        default=1,
        type=int,
        help="The number of GPUs for training. (default: 1)",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="The number of workers for dataloader. (default: 4)",
    )
    parser.add_argument(
        "--loss",
        default="si_snr",
        type=str,
        choices=["si_snr", "snr"],
        help="The loss function for network training, either snr or si_snr. (default: si_snr)",
    )
    parser.add_argument(
        "--mixture-residual",
        default="pass",
        type=str,
        choices=["all", "pass", "music_sfx"],
        help="Whether to add the residual to estimates, 'pass' doesn't add residual, 'all' splits residual among "
        "all sources, 'music_sfx' splits residual among only music and sfx sources . (default: pass)",
    )

    args = parser.parse_args()
    si_loss = True if args.loss == "si_snr" else False
    model = CocktailForkModule(si_loss=si_loss, mixture_residual=args.mixture_residual)
    train_loader, valid_loader, test_loader = _get_dataloaders(
        args.root_dir,
        args.train_batch_size,
        args.chunk_size,
        args.eval_batch_size,
        args.num_workers,
    )

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=5, verbose=True)
    callbacks = [checkpoint]
    if args.num_gpu > 0:
        devices = args.num_gpu
        accelerator = "gpu"
    else:
        devices = "auto"
        accelerator = "cpu"

    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        devices=devices,
        accelerator=accelerator,
        gradient_clip_val=5.0,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, valid_loader)
    model.load_from_checkpoint(checkpoint.best_model_path)
    ckpt = torch.load(checkpoint.best_model_path, map_location="cpu")
    model_weights = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    torch.save(model_weights, Path(checkpoint.dirpath) / "best_model.pth")
    trainer.test(model, test_loader)


if __name__ == "__main__":
    cli_main()
