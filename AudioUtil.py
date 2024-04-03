# This file was created with the help of the following tutorial:
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

import math
import random
import torch
import torchaudio
from matplotlib import pyplot as plt
from torchaudio import transforms
from IPython.display import Audio


class AudioUtil:
    @staticmethod
    def open(audio_file):
        """
        Load an audio file. Return the signal as a tensor and the sample rate
        :param audio_file: the path to the audio file
        :return: a tuple containing the signal and the sample rate
        """
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def rechannel(aud, new_channel):
        """
        Convert the given audio to the desired number of channels
        :param aud: the audio
        :param new_channel: the desired number of channels
        :return: the audio with the desired number of channels
        """
        sig, sr = aud

        if sig.shape[0] == new_channel:
            # Nothing to do
            return aud

        if new_channel == 1:
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return resig, sr

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        """
        Resample the given audio to the new sample rate
        :param aud: the audio
        :param newsr: the new sample rate
        :return: the resampled audio
        """
        sig, sr = aud

        if sr == newsr:
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return resig, newsr

    @staticmethod
    def pad_trunc(aud, max_ms):
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
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return sig, sr

    @staticmethod
    def time_shift(aud, shift_limit):
        """
        Shift the signal to the left or right by some percent. Values at the end are 'wrapped around' to the start of
        the transformed signal.
        :param aud: the audio
        :param shift_limit: the limit of the shift
        :return: the audio with the shifted signal
        """
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        """
        Create a spectrogram from a raw audio signal
        :param aud: the audio
        :param n_mels: the number of mel bins
        :param n_fft: the length of the windowed signal after padding with zeros
        :param hop_len: the number of samples between successive frames
        :return: the spectrogram
        """
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        """
        Augment a spectrogram by masking out some sections of it in both the frequency and time dimension
        :param spec: the spectrogram
        :param max_mask_pct: the maximum percentage of the spectrogram to mask out
        :param n_freq_masks: the number of frequency masks to apply
        :param n_time_masks: the number of time masks to apply
        :return: the augmented spectrogram
        """
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
