from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.util.AudioUtil import AudioUtil


class SoundDS(Dataset):
    def __init__(self, df, params, device):
        self.df = df
        self.device = device
        self.duration = 30000
        self.sr = 44100
        self.channel = 2

        if params is None:
            params = {}

        default_params = {
            "n_mels": 64,
            "normalize": True,
        }

        for key in default_params.keys():
            if key not in params:
                params[key] = default_params[key]

        self.params = params

    def __len__(self):
        return len(self.df)

    def _compute(self, audio_file) -> torch.Tensor:
        sig, sr = AudioUtil.open(audio_file)
        sig = sig.to(self.device)
        audio = AudioUtil.resample((sig, sr), self.sr)
        audio = AudioUtil.rechannel(audio, self.channel)
        audio = AudioUtil.pad_trunc(audio, self.duration)
        spectrogram = AudioUtil.spectrogram(audio, n_fft=1024, hop_len=None, n_mels=self.params["n_mels"],
                                            normalize=self.params["normalize"])
        return spectrogram

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_file = self.df.loc[idx, 'file_path']
        augmented_spectrogram = self._compute(audio_file)
        data = self.df.iloc[idx, 1:].values
        label = torch.Tensor(data.astype(int))
        return augmented_spectrogram, label.to(self.device)
