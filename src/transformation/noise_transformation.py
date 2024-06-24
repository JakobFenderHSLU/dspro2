import torch


class NoiseTransformation(torch.nn.Module):
    def __init__(self, noise_factor: float = 0.5):
        super().__init__()
        self.noise_factor = noise_factor

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(spectrogram)
        return spectrogram + self.noise_factor * noise
