import gc
import time

import pandas as pd
import torch
import torchaudio
import wandb
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.dataset.audio_dataset import AudioDataset
from src.scratch.classifier import AudioClassifier
from src.transformation.loop_trunk_transformation import LoopTrunkTransformation
from src.transformation.noise_transformation import NoiseTransformation
from src.transformation.normalize_transformation import NormalizeTransformation
from src.transformation.rechannel_transformation import RechannelTransformation
from src.transformation.resample_transformation import ResampleTransformation
from src.util.logger_utils import init_logging
from src.util.wandb_utils import WandbUtils

log = init_logging("scratch_runner")


class CnnFromScratchRunner:
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, scale: str = "debug") -> None:
        self.train_df: pd.DataFrame = train_df
        self.val_df: pd.DataFrame = val_df
        self.train_dl: DataLoader = None
        self.val_dl: DataLoader = None
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: AudioClassifier = None
        self.loss_function: CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device)
        self.scale: str = scale
        self.best_f1 = 0
        self.best_f1_epoch = 0

        self.wandb_utils = None

        self.class_labels = ["_".join(self.train_df.columns[i].split("_")[2:])
                             for i in range(1, len(self.train_df.columns))]
        self.class_counts = len(self.train_df.columns) - 1

    def run(self) -> None:
        formatted_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        wandb_run_name = f"{formatted_time}-scratch-{self.scale}-{self.class_counts}"

        sweep_config: dict = {
            "name": wandb_run_name,
            "method": "grid",
            "metric": {"goal": "maximize", "name": "val/f1"},
            "parameters": {
                "epochs": {"value": 5000},
                "learning_rate": {"values": [0.0001, 0.00001]},  # {"min": 0.0001, "max": 0.1},
                "batch_size_train": {"values": [16]},  # try higher
                "batch_size_val": {"values": [16]},  # try higher
                "anneal_strategy": {"values": ["linear"]},
                "weight_decay": {"values": [0, 0.001]},

                # Parameters for SpectrogramPipeline
                "sample_rate": {"values": [44_100]},
                "channel": {"values": [1]},
                "duration_ms": {"values": [15_000]},
                "n_mels": {"values": [64]},  # {"min": 32, "max": 128},
                "n_fft": {"values": [1024]},
                "normalize": {"values": [True]},  # Through testing, normalization is better
                "noise_factor": {"values": [0, 0.1]},
            },
            "optimizer": ["adam"],
        }

        log.debug(f"Starting sweep with config:")
        log.debug(sweep_config)
        sweep_id: str = wandb.sweep(sweep=sweep_config, project="cnn_from_scratch", entity="swiss-birder")
        log.debug("started sweep with id: {sweep_id}")
        log.info(f"Using device {self.device}")
        wandb.agent(sweep_id, function=self._run, count=100)

    def _run(self) -> None:
        formatted_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        wandb_run_name = f"{formatted_time}-scratch-{self.scale}-{self.class_counts}"
        self.run = wandb.init(name=wandb_run_name, project="cnn_from_scratch", entity="swiss-birder")

        self.best_f1 = 0
        self.best_f1_epoch = 0

        self.wandb_utils = WandbUtils(self.run, self.class_labels)

        transformations = torch.nn.Sequential(
            ResampleTransformation(sample_rate=wandb.config.sample_rate),
            RechannelTransformation(channel=wandb.config.channel),
            LoopTrunkTransformation(duration_ms=wandb.config.duration_ms, sample_rate=wandb.config.sample_rate),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=wandb.config.sample_rate,
                n_mels=wandb.config.n_mels,
                n_fft=wandb.config.n_fft
            ),
            NormalizeTransformation(),
            NoiseTransformation(noise_factor=wandb.config.noise_factor)
        )

        train_ds = AudioDataset(self.train_df, transform=transformations, duration_ms=wandb.config.duration_ms)
        val_ds = AudioDataset(self.val_df, transform=transformations, duration_ms=wandb.config.duration_ms)

        log.debug(f"Train DS length: {len(train_ds)}")
        log.debug(f"Val DS length: {len(val_ds)}")

        self.train_dl = DataLoader(train_ds,
                                   batch_size=wandb.config.batch_size_train,
                                   num_workers=8,
                                   prefetch_factor=2,
                                   persistent_workers=True)
        self.val_dl = DataLoader(val_ds,
                                 batch_size=wandb.config.batch_size_val,
                                 num_workers=8,
                                 prefetch_factor=2,
                                 persistent_workers=True)

        self.model = AudioClassifier(self.class_counts, in_channels=wandb.config.channel).to(self.device)

        img = wandb.Image(self.val_dl.dataset[0][0][0].cpu().numpy())
        wandb.log({"debug/img": img})

        wandb.watch(self.model, log="all")
        self._training()
        wandb.unwatch(self.model)

        wandb.finish()

    def _training(self) -> None:

        self.model.train()

        log.info("Starting Training")
        # Loss Function, Optimizer and Scheduler
        optimizer: Optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=wandb.config.learning_rate,
            weight_decay=wandb.config.weight_decay
        )
        scheduler: OneCycleLR = OneCycleLR(optimizer, max_lr=wandb.config.learning_rate,
                                           steps_per_epoch=int(len(self.train_dl)),
                                           epochs=wandb.config.epochs,
                                           anneal_strategy=wandb.config.anneal_strategy)

        is_first = True
        # Repeat for each epoch
        for epoch_iter in range(wandb.config.epochs):
            epoch = epoch_iter + 1
            log.info(f'Epoch: {epoch}')
            epoch_time = time.time()

            y_true = []
            y_pred_probabilities = []

            # Repeat for each batch in the training set
            test_time = time.time()
            is_first_in_epoch = True
            for data in self.train_dl:
                if test_time:
                    log.info(f"Time to get batch: {time.time() - test_time}")
                    test_time = None
                timestamp = time.time()
                # Get the input features and target labels, and put them on the GPU
                inputs: Tensor = data[0].to(self.device)
                labels: Tensor = data[1].to(self.device)

                for i in range(inputs.size(0)):
                    if inputs[i].sum() == 0:
                        log.error(f"Tensor with label {labels[i]} is empty")

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                predictions: Tensor = self.model(inputs)
                loss: Tensor = self.loss_function(predictions, labels)

                if is_first:
                    log.debug("pred")
                    log.debug(predictions[0])
                    log.debug("loss")
                    log.debug(loss)

                y_pred_probabilities.extend(predictions.detach().cpu().numpy())
                y_true.extend(labels.cpu().numpy())

                loss.backward()
                optimizer.step()
                scheduler.step()

                is_first = False

                if epoch_iter % 10 == 0:
                    if is_first_in_epoch:
                        wandb.log({
                            "debug/train_loss": loss.item(),
                            "debug/time_per_batch": time.time() - timestamp
                        })
                        is_first_in_epoch = False
            gc.collect()

            self.wandb_utils.log_model_results(y_true, y_pred_probabilities, mode="train")

            # save model to wandb
            model_name = f"./output/{self.run.name}_{epoch}.onnx"
            torch.onnx.export(self.model, torch.randn(1, wandb.config.channel, 64, 64).to(self.device), model_name)

            log.info(f"Model saved to {model_name}")
            wandb.save(model_name)

            self._inference(epoch)
            gc.collect()
            epoch_duration = time.time() - epoch_time
            log.info(f"Epoch time: {int(epoch_duration // 60)} min {int(epoch_duration % 60)} sec")

            if epoch > 200 and epoch - self.best_f1_epoch > 100:
                log.info(f"Stopping early at epoch {epoch}")
                wandb.log({"debug/early_stop": True})
                break

        log.info('Finished Training')

    def _inference(self, epoch: int) -> None:

        self.model.eval()

        # Disable gradient updates
        with (torch.no_grad()):
            y_pred_probabilities = []
            y_true = []

            is_first = True
            for data in self.val_dl:
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

            new_f1 = self.wandb_utils.log_model_results(y_true, y_pred_probabilities, mode="val")

            if new_f1 >= self.best_f1:
                self.best_f1 = new_f1
                self.best_f1_epoch = epoch
                log.info(f"F1: {new_f1} (new best)")
            else:
                log.info(f"F1: {new_f1}")

        log.info('Finished Inference')
