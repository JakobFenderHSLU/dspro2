import torch.nn


class RechannelTransformer(torch.nn.Module):
    def __init__(self, channel: int = 2):
        super().__init__()
        self.channel = channel

    def forward(self, signal: torch.Tensor) -> torch.Tensor:

        if signal.shape[0] == self.channel:
            # Nothing to do
            return signal

        if self.channel == 1:
            # Convert from stereo to mono by selecting only the first channel
            ret_signal = signal[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            ret_signal = torch.cat([signal, signal])

        return ret_signal
