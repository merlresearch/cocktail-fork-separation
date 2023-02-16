<!--
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: MIT
-->
# Baseline Model for the Cocktail Fork Problem

This repository includes source code for training and using the Multi Resolution CrossNet (MRX) model proposed in our ICASSP 2022 paper,
**The Cocktail Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks**
by Darius Petermann, Gordon Wichern, Zhong-Qiu Wang, and Jonathan Le Roux.

[Please click here to read the paper.](https://arxiv.org/pdf/2110.09958.pdf)

If you use any part of this code for your work, we ask that you include the following citation:

    @InProceedings{Petermann2022ICASSP05,
      author =	 {Petermann, Darius and Wichern, Gordon and Wang, Zhong-Qiu and {Le Roux}, Jonathan},
      title =	 {The Cocktail Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks},
      booktitle =	 {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year =	 2022,
      month =	 may
    }


## Table of contents

1. [Environment Setup](#environment-setup)
2. [Using a pre-trained model](#using-a-pre-trained-model)
3. [Training a model on the Divide and Remaster Dataset](#training-a-model-on-the-divide-and-remaster-dataset)
4. [Evaluating a model on the Divide and Remaster Dataset](#evaluating-a-model-on-the-divide-and-remaster-dataset)
5. [License](#license)

## Environment Setup

The code has been tested using `python 3.9`. Necessary dependencies can be installed using the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you prefer to use the [torchaudio soundfile backend](https://pytorch.org/audio/stable/backend.html) (required on windows) please refer to the [SoundFile documentation](https://pysoundfile.readthedocs.io/en/latest/) for installation instructions.

Please modify pytorch installation depending on your particular CUDA version if necessary.

## Using a pre-trained model

To separate a soundtrack (e.g., movie or TV commercial), we include via git LFS a pre-trained model,
which can be used from the command line as:

```bash
python separate.py --audio-path /input/soundtrack.wav --out-dir /separated/track1
```

and will save `speech.wav`, `music.wav`, and `sfx.wav` in `out-dir`.

Alternatively, inside of python we can separate a torch.Tensor of shape `(channels, n_samples)`, at a sampling rate of 44.1 kHz:

```python
import separate
enhanced_dict = separate.separate_soundtrack(audio_tensor, ...)
```

It is also possible to use a model you trained:

```python
import torch
import separate
from mrx import MRX
my_model = MRX(**kwargs).eval()
state_dict = torch.load(MY_TRAINED_MODEL_PATH)
my_model.load_state_dict(state_dict)
enhanced_dict = separate.separate_soundtrack(audio_tensor, separation_model=my_model, ...)
```

We include two pre-trained models in the `checkpoints` directory:
1. `default_mrx_pre_trained_weights.pth`: This is the model trained using the default arguments from [`lightning_train.py`](./lightning_train.py), except the training loss is SNR (`--loss snr`). This ensures that the level of the output signals matches the mixture.
2. `paper_mrx_pre_trained_weights.pth`: This is the model trained using the default arguments from [`lightning_train.py`](./lightning_train.py) including scale-invariant SNR loss function, which reproduces the results from our ICASSP paper.
However, due to the scale-invariant training the level of the output signals will not match the mixture.

## Training a model on the Divide and Remaster Dataset

If you haven't already, you will first need to download the [Divide and Remaster (DnR) Dataset.](https://zenodo.org/record/6949108#.Y861fOLMKrN)

We provide [`lightning_train.py`](./lightning_train.py) for model training, using:

```bash
python lightning_train.py \
        [--root-dir DNR_ROOT_DIR] \
        [--num-gpu NUM_GPU] \
        [--num-workers NUM_WORKERS] \
        ...
```
where `DNR_ROOT_DIR` is the top level DnR directory,  containing the `tr`, `cv`, and `tt` folders.\
Details of other parameter values can be found using:

```bash
python lightning_train.py --help
```

## Evaluating a model on the Divide and Remaster Dataset

To evaluate the scale-invariant source to distortion ratio (SI-SDR) on the DnR test set using a pre-trained model:

```bash
python eval.py \
        [--root-dir DNR_ROOT_DIR] \
        [--checkpoint CHECKPOINT] \
        [--gpu-device GPU_DEVICE] \
```

The following is the average SI-SDR (dB) of the DnR test set using the included pre-trained model, which was trained using the default configuration of `lightning_train.py`.

|             | Speech | Music |  SFX |
|:------------|-------:|------:|-----:|
| Unprocessed |    1.0 |  -6.8 | -5.0 |
| MRX (repo)  |   12.5 |   4.2 |  5.7 |
| MRX (paper) |   12.3 |   4.2 |  5.7 |


## License

Released under `MIT` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:
```
Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: MIT
```
