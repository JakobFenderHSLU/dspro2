import time

import pandas as pd
import torch
import torchaudio
import wandb
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.dataset.audio_dataset import AudioDataset
from src.transformation.loop_trunk_transformation import LoopTrunkTransformation
from src.transformation.normalize_transformation import NormalizeTransformation
from src.transformation.rechannel_transformation import RechannelTransformation
from src.transformation.resample_transformation import ResampleTransformation
from src.util.logger_utils import init_logging
from src.util.wandb_utils import WandbUtils

log = init_logging("scratch_test")


class CnnFromScratchTester:

    def __init__(self, model_path: str, test_df: pd.DataFrame):
        self.model_path = model_path
        self.test_df = test_df
        self.test_dl = None
        self.model = None
        self.run = None
        self.wandb_utils = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function: CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device)

        self.class_labels = ["_".join(self.test_df.columns[i].split("_")[2:])
                             for i in range(1, len(self.test_df.columns))]
        self.class_counts = len(test_df.columns) - 1

    def test(self):
        formatted_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        wandb_run_name = f"{formatted_time}-scratch-test-{self.class_counts}"
        self.run = wandb.init(name=wandb_run_name, project="cnn_from_scratch", entity="swiss-birder")
        self.wandb_utils = WandbUtils(self.run, self.class_labels)

        self.model = torch.load(self.model_path)
        self.model.eval()
        self.model.to(self.device)

        transformations = torch.nn.Sequential(
            ResampleTransformation(sample_rate=44_100),
            RechannelTransformation(channel=1),
            LoopTrunkTransformation(duration_ms=15, sample_rate=44_100),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=44_100,
                n_mels=64,
                n_fft=1024,
            ),
            NormalizeTransformation()
        )

        test_ds = AudioDataset(self.test_df, transform=transformations, duration_ms=15)

        self.test_dl = DataLoader(test_ds,
                                  batch_size=16,
                                  num_workers=8,
                                  prefetch_factor=2,
                                  persistent_workers=True)

        wandb.watch(self.model, log="all")

        self.model.eval()

        # Disable gradient updates
        with (torch.no_grad()):
            y_pred_probabilities = []
            y_true = []

            is_first = True
            for data in self.test_dl:
                # Get the input features and target labels, and put them on the GPU
                inputs: Tensor = data[0].to(self.device)
                labels: Tensor = data[1].to(self.device)

                if is_first:
                    log.debug(f"First entry of batch: ")
                    log.debug(f"input {inputs.size()}")
                    log.debug(inputs[0])
                    log.debug(f"label {labels.size()}")
                    log.debug(labels[0])

                for i in range(inputs.size(0)):
                    if inputs[i].sum() == 0:
                        log.error(f"Tensor with label {labels[i]} is empty")

                # Get predictions
                predictions: Tensor = self.model(inputs)
                loss: Tensor = self.loss_function(predictions, labels)

                wandb.log({"debug/val_loss": loss.item()})

                y_true.extend(labels.cpu().numpy())
                y_pred_probabilities.extend(predictions.cpu().numpy())

            self.wandb_utils.log_model_results(y_true, y_pred_probabilities, mode="test")

        log.info('Finished Inference')
