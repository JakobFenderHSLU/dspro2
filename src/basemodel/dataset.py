from typing import Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from src.util.audio_utils import AudioUtil


class SoundDS(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = torchaudio.load(self.df.loc[idx, 'file_path'])
        label = torch.Tensor(self.df.iloc[idx, 1:].values.astype(int))

        spectrogram = self.transform(audio)

        return spectrogram, label
