import time
from typing import Tuple

import h5py
import torch
from torch.utils.data import Dataset

from src.util.AudioUtil import AudioUtil


class SoundDS(Dataset):
    def __init__(self, df, device):
        self.df = df
        self.device = device
        self.duration = 30000
        self.sr = 44100
        self.channel = 2
        # self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)

    def _compute(self, audio_file) -> torch.Tensor:
        sig, sr = AudioUtil.open(audio_file)
        sig = sig.to(self.device)
        audio = AudioUtil.resample((sig, sr), self.sr)
        audio = AudioUtil.rechannel(audio, self.channel)
        audio = AudioUtil.pad_trunc(audio, self.duration)
        # audio = AudioUtil.time_shift(audio, self.shift_pct)
        spectrogram = AudioUtil.spectro_gram(audio, n_mels=64, n_fft=1024, hop_len=None)
        # spectrogram = AudioUtil.spectro_augment(spectrogram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return spectrogram

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_file = self.df.loc[idx, 'file_path']
        augmented_spectrogram = self._compute(audio_file)
        data = self.df.iloc[idx, 1:].values
        label = torch.Tensor(data.astype(int))
        return augmented_spectrogram, label.to(self.device)

