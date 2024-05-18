from torch import Tensor
from torchaudio_augmentations import Gain, Noise, PolarityInversion, LowPassFilter, Delay


class AugmentationUtils:
    def __init__(self, sr: int):
        self.sr = sr

    @staticmethod
    def gain(self, sig: Tensor, min_gain: float, max_gain: float) -> Tensor:
        sig = Gain(min_gain=min_gain, max_gain=max_gain)(sig)
        return sig

    @staticmethod
    def static_noise(self, sig: Tensor, min_snr: float, max_snr: float) -> Tensor:
        sig = Noise(min_snr, max_snr)(sig)
        return sig

    @staticmethod
    def polarity_inversion(self, sig: Tensor) -> Tensor:
        sig = PolarityInversion()(sig)
        return sig

    @staticmethod
    def low_pass_filter(self, sig: Tensor, min_cutoff_freq: float, max_cutoff_freq: float) -> Tensor:
        sig = LowPassFilter(self.sr, min_cutoff_freq, max_cutoff_freq)(sig.unsqueeze(0)).squeeze(0)
        return sig

    @staticmethod
    def delay(self, sig: Tensor, min_delay: int, max_delay: int) -> Tensor:
        sig = Delay(self.sr, min_delay, max_delay)(sig)
        return sig
