import torch


class NormalizeTransformation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # log(1 + x) -> min-max scaling
        # see https://stackoverflow.com/questions/72785857/normalize-a-melspectrogram-to-0-255-with-or-without-frequency-scaling
        spectrogram = torch.log1p(spectrogram)
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        return spectrogram
