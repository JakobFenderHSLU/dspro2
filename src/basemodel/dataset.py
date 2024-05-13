import time

import wandb
from torch.utils.data import Dataset

from src.util.AudioUtil import AudioUtil


class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df
        self.duration = 30000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Do all the audio augmentation here
        timestamp = time.time()
        audio_file = self.df.loc[idx, 'file_path']
        class_id = self.df.loc[idx, 'species_id']

        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        wandb.log({"data_loading_time": time.time() - timestamp})

        return aug_sgram, class_id
