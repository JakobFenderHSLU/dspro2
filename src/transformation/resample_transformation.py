from typing import Tuple

import torch
import torchaudio


class ResampleTransformation(torch.nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, audio: Tuple[torch.Tensor, int]) -> torch.Tensor:
        sig, sr = audio

        if sr == self.sample_rate:
            # Nothing to do
            return sig

        num_channels = sig.shape[0]
        # Resample first channel
        resampled_signal = torchaudio.transforms.Resample(sr, self.sample_rate).to(sig.device)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            resampled_signal_2 = torchaudio.transforms.Resample(sr, self.sample_rate).to(sig.device)(sig[1:, :])
            resampled_signal = torch.cat([resampled_signal, resampled_signal_2])

        return resampled_signal
