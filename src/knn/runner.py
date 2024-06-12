import gc

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from src.dataset.audio_dataset import AudioDataset
from src.feature_extractor.beat_embedder import BeatEmbedder
from src.transformation.loop_trunk_transformation import LoopTrunkTransformation
from src.transformation.rechannel_transformation import RechannelTransformation
from src.transformation.resample_transformation import ResampleTransformation
from src.util.logger_utils import init_logging

log = init_logging("KNN Runner")


class KnnRunner:
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.train_df: pd.DataFrame = train_df
        self.val_df: pd.DataFrame = val_df
        self.train_dl: DataLoader = None
        self.val_dl: DataLoader = None
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: KNeighborsClassifier = None
        self.feature_extractor = BeatEmbedder(self.device)

    @property
    def class_count(self) -> int:
        return len(self.train_df.columns) - 1

    def run(self) -> None:
        self._run()

    def _run(self):
        transformations = torch.nn.Sequential(
            ResampleTransformation(sample_rate=16_000),  # BEATs expects 16'000 Hz
            RechannelTransformation(channel=1),
            LoopTrunkTransformation(duration_ms=30_000, sample_rate=16_000),
        )

        train_ds = AudioDataset(self.train_df, transform=transformations, duration_ms=30_000)
        val_ds = AudioDataset(self.val_df, transform=transformations, duration_ms=30_000)

        self.train_dl = DataLoader(train_ds,
                                   batch_size=8,
                                   num_workers=4,
                                   prefetch_factor=2,
                                   persistent_workers=True)
        self.val_dl = DataLoader(val_ds,
                                 batch_size=8,
                                 num_workers=4,
                                 prefetch_factor=2,
                                 persistent_workers=True)

        self.model = KNeighborsClassifier(n_neighbors=self.class_count)

        self._train()
        gc.collect()
        self._eval()

    def _train(self) -> None:
        for data in self.train_dl:
            x, y = data

            log.debug(f"Shape of x: {x.shape}")
            log.debug(f"Shape of y: {y.shape}")

            x = x.to(self.device)
            x = [self.feature_extractor.embed(x_i) for x_i in x]
            x = [x_i.cpu().detach().numpy() for x_i in x]
            # current shape (8, 1496,768) -> (8, *)
            x = [x_i.flatten() for x_i in x]

            # onehot encode to number
            y = torch.argmax(y, dim=1)
            y = y.numpy()
            self.model.fit(x, y)
            break

    def _eval(self) -> None:
        y_preds = []
        y_actual = []

        for data in self.val_dl:
            x, y = data

            x = x.to(self.device)
            x = [self.feature_extractor.embed(x_i) for x_i in x]
            x = [x_i.cpu().detach().numpy() for x_i in x]
            x = [x_i.flatten() for x_i in x]

            y = torch.argmax(y, dim=1)
            y = y.numpy()
            y_pred = [self.model.predict([x_i])[0] for x_i in x]

            y_preds.extend(y_pred)
            y_actual.extend(y)

        gc.collect()

        accuracy = accuracy_score(y_actual, y_preds)
        f1 = f1_score(y_actual, y_preds, average='weighted')

        log.info(f"Accuracy: {accuracy}")
        log.info(f"F1: {f1}")
        log.info(f"Confusion Matrix: {pd.crosstab(pd.Series(y_actual), pd.Series(y_preds))}")
