import gc
import time

import numpy as np
import pandas as pd
import torch
import torchaudio
import wandb
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.dataset.audio_dataset import AudioDataset
from src.transformation.loop_trunk_transformation import LoopTrunkTransformation
from src.transformation.normalize_transformation import NormalizeTransformation
from src.transformation.rechannel_transformation import RechannelTransformation
from src.transformation.resample_transformation import ResampleTransformation
from src.util.logger_utils import init_logging
from src.vggish.classifier import VGGishClassifier

log = init_logging("VGGish_runner")


class VGGishRunner(object):
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, scale: str = "debug") -> None:
        self.train_df: pd.DataFrame = train_df
        self.val_df: pd.DataFrame = val_df
        self.train_dl: DataLoader = None
        self.val_dl: DataLoader = None
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: VGGishClassifier = None
        self.loss_function: CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device)
        self.scale: str = scale

    @property
    def class_counts(self) -> int:
        return len(self.train_df.columns) - 1

    def run(self) -> None:
        formatted_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        wandb_run_name = f"{formatted_time}-VGGish-{self.scale}-{self.class_counts}"

        sweep_config: dict = {
            "name": wandb_run_name,
            "method": "grid",
            "metric": {"goal": "maximize", "name": "val/f1"},
            "parameters": {
                "epochs": {"value": 1000},
                "learning_rate": {"value": 0.1},  # {"min": 0.0001, "max": 0.1},
                "batch_size_train": {"values": [16]},  # try higher
                "batch_size_val": {"values": [16]},  # try higher
                "anneal_strategy": {"values": ["linear"]},

                # Parameters for SpectrogramPipeline
                "sample_rate": {"values": [16_000]},
                "channel": {"values": [1]},
                "duration_ms": {"values": [30_000]},
                "n_mels": {"values": [64]},  # {"min": 32, "max": 128},
                "n_fft": {"values": [1024]},
                "normalize": {"values": [True]},  # Through testing, normalization is better
            },
            "optimizer": ["adam"],
        }

        log.debug(f"Starting sweep with config:")
        log.debug(sweep_config)
        sweep_id: str = wandb.sweep(sweep=sweep_config, project="VGGish", entity="swiss-birder")
        log.debug("started sweep with id: {sweep_id}")
        log.info(f"Using device {self.device}")
        wandb.agent(sweep_id, function=self._run, count=100)


    def _run(self) -> None:
        formatted_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        wandb_run_name = f"{formatted_time}-scratch-{self.scale}-f{self.class_counts}"
        run = wandb.init(name=wandb_run_name, project="cnn_from_scratch", entity="swiss-birder")

        transformations = torch.nn.Sequential(
            ResampleTransformation(sample_rate=wandb.config.sample_rate),
            RechannelTransformation(channel=wandb.config.channel),
            LoopTrunkTransformation(duration_ms=wandb.config.duration_ms, sample_rate=wandb.config.sample_rate),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=wandb.config.sample_rate,
                n_mels=wandb.config.n_mels,
                n_fft=wandb.config.n_fft
            ),
            NormalizeTransformation()
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

        self.model = VGGishClassifier(self.class_counts).to(self.device)

        img = wandb.Image(self.val_dl.dataset[0][0][0].cpu().numpy())
        wandb.log({"debug/img": img})

        wandb.watch(self.model, log="all")
        self._training()
        wandb.unwatch(self.model)

        model_name = f"./output/{run.id}_{run.name}_model.onnx"
        torch.onnx.export(self.model, torch.randn(1, 2, 64, 64).to(self.device), model_name)

        log.info(f"Model saved to {model_name}")
        wandb.save(model_name)
        wandb.finish()

    def _training(self) -> None:
        log.info("Starting Training")
        # Loss Function, Optimizer and Scheduler
        optimizer: Optimizer = torch.optim.Adam(self.model.parameters(), lr=wandb.config.learning_rate)
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

            # Repeat for each batch in the training set
            test_time = time.time()
            is_first_in_epoch = True
            for data in self.train_dl:  # very slow
                if test_time:
                    log.info(f"Time to get batch: {time.time() - test_time}")
                    test_time = None
                timestamp = time.time()
                # Get the input features and target labels, and put them on the GPU
                inputs: Tensor = data[0].to(self.device)
                labels: Tensor = data[1].to(self.device)

                # first entry of batch
                if is_first:
                    log.debug(f"First entry of batch: ")
                    log.debug("input")
                    log.debug(inputs[0])
                    log.debug("label")
                    log.debug(labels[0])

                for i in range(inputs.size(0)):
                    if inputs[i].sum() == 0:
                        log.error(f"Tensor with label {labels[i]} is empty")

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                preds: Tensor = self.model(inputs)
                loss: Tensor = self.loss_function(preds, labels)

                if is_first:
                    log.debug("pred")
                    log.debug(preds[0])
                    log.debug("loss")
                    log.debug(loss)

                loss.backward()
                optimizer.step()
                scheduler.step()

                is_first = False

                if epoch_iter % 10 == 0:
                    if is_first_in_epoch:
                        wandb.log({
                            "debug/train/loss": loss.item(),
                            "debug/time_per_batch": time.time() - timestamp
                        })
                        is_first_in_epoch = False
            gc.collect()
            self._inference()
            gc.collect()
            epoch_duration = time.time() - epoch_time
            log.info(f"Epoch time: {int(epoch_duration // 60)} min {int(epoch_duration % 60)} sec")
        log.info('Finished Training')

    def _inference(self) -> None:
        log.info("Starting Inference")
        # Disable gradient updates
        with (torch.no_grad()):
            y_pred = []
            y_true = []

            is_first = True
            for data in self.val_dl:
                # Get the input features and target labels, and put them on the GPU
                inputs: Tensor = data[0].to(self.device)
                labels: Tensor = data[1].to(self.device)

                if is_first:
                    log.debug(f"First entry of batch: ")
                    log.debug("input")
                    log.debug(inputs[0])
                    log.debug("label")
                    log.debug(labels[0])

                for i in range(inputs.size(0)):
                    if inputs[i].sum() == 0:
                        log.error(f"Tensor with label {labels[i]} is empty")

                y_true.extend(labels.cpu().numpy())

                # Get predictions
                preds: Tensor = self.model(inputs)
                loss: Tensor = self.loss_function(preds, labels)

                if is_first:
                    log.debug("pred")
                    log.debug(preds[0])
                    log.debug("loss")
                    log.debug(loss)

                # Count of predictions that matched the target label
                y_pred.extend(preds.cpu().numpy())

            # y_pred make the highest value 1 and the rest 0 for this np array
            y_pred_argmax = np.argmax(y_pred, axis=1)
            new_y_pred = np.zeros_like(y_pred)
            new_y_pred[np.arange(len(y_pred)), y_pred_argmax] = 1

            y_pred = new_y_pred

            if is_first:
                log.debug("y_true")
                log.debug(y_true[0])
                log.debug("y_pred (argmax)")
                log.debug(y_pred[0])

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            log.info(f"Accuracy: {acc}")
            log.info(f"F1: {f1}")

            wandb.log({
                "val/accuracy": acc,
                "val/f1": f1
            })
            # check if sum of y_true is 0

            # confusion matrix
            # onehot to label
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

            if is_first:
                log.debug("y_true (argmax)")
                log.debug(y_true[0])
                log.debug("y_pred (argmax)")
                log.debug(y_pred[0])

            # create confusion matrix
            wandb.log({"debug/confusion_matrix": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred)})

        log.info('Finished Inference')
