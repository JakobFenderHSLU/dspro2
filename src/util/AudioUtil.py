# This file was created with the help of the following tutorial:
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
import random

import torch
import torchaudio
from torch import Tensor
from torchaudio import transforms


class AudioUtil:
    @staticmethod
    def open(audio_file: str) -> (torch.Tensor, int):
        """
        Load an audio file. Return the signal as a tensor and the sample rate
        :param audio_file: the path to the audio file
        :return: a tuple containing the signal and the sample rate
        """
        # Note (Jakob): This outputs a lot of useless information and spams the console.
        # I have not found a way to suppress it.
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def rechannel(audio, new_channel) -> (torch.Tensor, int):
        """
        Convert the given audio to the desired number of channels
        :param audio: the audio
        :param new_channel: the desired number of channels
        :return: the audio with the desired number of channels
        """
        signal, sample_rate = audio

        if signal.shape[0] == new_channel:
            # Nothing to do
            return audio

        if new_channel == 1:
            # Convert from stereo to mono by selecting only the first channel
            mono_signal = signal[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            mono_signal = torch.cat([signal, signal])

        return mono_signal, sample_rate

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, new_sr) -> (torch.Tensor, int):
        """
        Resample the given audio to the new sample rate
        :param aud: the audio
        :param new_sr: the new sample rate
        :return: the resampled audio
        """
        sig, sr = aud

        if sr == new_sr:
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, new_sr).to(sig.device)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, new_sr).to(sig.device)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return resig, new_sr

    @staticmethod
    def pad_trunc(aud, max_ms) -> (torch.Tensor, int):
        """
        Pad or truncate the signal to a fixed length
        :param aud: the audio
        :param max_ms: the desired length in milliseconds
        :return: the audio with the desired length
        """
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len)).to(sig.device)
            pad_end = torch.zeros((num_rows, pad_end_len)).to(sig.device)

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return sig, sr

    @staticmethod
    def time_shift(audio, shift_limit) -> (torch.Tensor, int):
        """
        Shift the signal to the left or right by some percent. Values at the end are 'wrapped around' to the start of
        the transformed signal.
        :param audio: the audio
        :param shift_limit: the limit of the shift
        :return: the audio with the shifted signal
        """
        signal, sampling_rate = audio
        _, signal_length = signal.shape
        shift_amt = int(random.random() * shift_limit * signal_length)
        return signal.roll(shift_amt), sampling_rate

    @staticmethod
    def spectrogram(aud: Tensor, n_mels: int = 64, n_fft: int = 1024, hop_len=None,
                    normalize: bool = True) -> torch.Tensor:
        """
        Create a spectrogram from a raw audio signal
        :param aud: the audio
        :param n_mels: the number of mel bins
        :param n_fft: the length of the windowed signal after padding with zeros
        :param hop_len: the number of samples between successive frames
        :return: the spectrogram
        """
        signal, sampling_rate = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec_transform = transforms.MelSpectrogram(sampling_rate,
                                                   n_fft=n_fft,
                                                   hop_length=hop_len,
                                                   n_mels=n_mels).to(signal.device)
        spectrogram = spec_transform(signal)

        if normalize:
            # log(1 + x)
            # see https://stackoverflow.com/questions/72785857/normalize-a-melspectrogram-to-0-255-with-or-without-frequency-scaling
            spectrogram = torch.log1p(spectrogram)

        # Convert to decibels
        spectrogram = transforms.AmplitudeToDB(top_db=top_db)(spectrogram)
        return spectrogram

    @staticmethod
    def spectro_augment(spectrogram: Tensor, max_mask_pct: float = 0.1, n_freq_masks: int = 1,
                        n_time_masks: int = 1) -> torch.Tensor:
        """
        Augment a spectrogram by masking out some sections of it in both the frequency and time dimension
        :param spectrogram: the spectrogram
        :param max_mask_pct: the maximum percentage of the spectrogram to mask out
        :param n_freq_masks: the number of frequency masks to apply
        :param n_time_masks: the number of time masks to apply
        :return: the augmented spectrogram
        """
        _, n_mels, n_steps = spectrogram.shape
        mask_value = spectrogram.mean()
        aug_spec = spectrogram

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
