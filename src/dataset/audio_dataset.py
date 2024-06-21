from typing import Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from src.util.logger_utils import init_logging

log = init_logging("dataset")


class AudioDataset(Dataset):
    def __init__(self, df, transform, duration_ms: int):
        self.df = df
        self.transform = transform
        self.duration_s = int(duration_ms / 1000)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = torchaudio.load(self.df.loc[idx, 'file_path'], num_frames=self.duration_s * 44_100)
        label = torch.Tensor(self.df.iloc[idx, 1:].values.astype(int))

        spectrogram = self.transform(audio)

        if spectrogram.isnan().any():
            log.error(f"File {self.df.loc[idx, 'file_path']} contains NaN values")

        return spectrogram, label
