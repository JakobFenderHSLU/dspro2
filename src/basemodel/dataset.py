import time
from typing import Tuple

import torch
import wandb
from torch import Tensor
from torch.utils.data import Dataset

from src.util.AudioUtil import AudioUtil


class SoundDS(Dataset):
    def __init__(self, df, device):
        self.df = df
        self.device = device
        self.duration = 30000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Tensor]:
        # Do all the audio augmentation here
        audio_file = self.df.loc[idx, 'file_path']
        class_id = self.df.loc[idx, 'species_id']

        sig, sr = AudioUtil.open(audio_file)
        sig = sig.to(self.device)
        audio = AudioUtil.resample((sig, sr), self.sr)
        audio = AudioUtil.rechannel(audio, self.channel)
        audio = AudioUtil.pad_trunc(audio, self.duration)
        audio = AudioUtil.time_shift(audio, self.shift_pct)
        spectrogram = AudioUtil.spectro_gram(audio, n_mels=64, n_fft=1024, hop_len=None)
        augmented_spectrogram = AudioUtil.spectro_augment(spectrogram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        # print size of augmented spectrogram
        print(augmented_spectrogram.size())
        raise Exception("Stop here")

        return augmented_spectrogram, class_id
