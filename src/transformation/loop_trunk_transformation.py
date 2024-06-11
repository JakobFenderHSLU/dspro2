import torch


class LoopTrunkTransformation(torch.nn.Module):
    def __init__(self, duration_ms: int, sample_rate: int):
        super().__init__()
        self.duration_ms = duration_ms
        self.sample_rate = sample_rate

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        max_len = int(self.sample_rate / 1000 * self.duration_ms)

        # repeat the audio until it reaches the desired length
        while signal.shape[1] < max_len:
            signal = torch.cat([signal, signal], 1)

        # Truncate
        sig = signal[:, :max_len]

        return sig
