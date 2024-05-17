import logging
import time

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import f1_score, accuracy_score
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.basemodel.classifier import AudioClassifier
from src.basemodel.dataset import SoundDS
from src.util.LoggerUtils import init_logging

log = init_logging("basemodel")


class BasemodelRunner:
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, verbose: bool) -> None:
        self.train_df: pd.DataFrame = train_df
        self.val_df: pd.DataFrame = val_df
        self.train_dl: DataLoader = None
        self.val_dl: DataLoader = None
        self.class_counts: int = len(train_df.columns) - 1
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: AudioClassifier = None
        self.loss_function: CrossEntropyLoss = nn.CrossEntropyLoss()

        if verbose:
            log.setLevel(logging.DEBUG)

    def run(self) -> None:
        sweep_config: dict = {
            "name": "Baseline Sweep",
            "method": "grid",
            "metric": {"goal": "maximize", "name": "val_f1"},
            "parameters": {
                "epochs": {"value": 10000},
                "learning_rate": {"value": 0.1},  # {"min": 0.0001, "max": 0.1},
                "batch_size_train": {"values": [64]},  # try higher
                "batch_size_val": {"values": [64]},  # try higher
                "anneal_strategy": {"values": ["linear"]},
                "n_mels": {"values": [64]},  # try
            }
        }
        log.debug(f"Starting sweep with config:")
        log.debug(sweep_config)
        sweep_id: str = wandb.sweep(sweep=sweep_config, project="Baseline-Full", entity="swiss-birder")
        log.debug("started sweep with id: {sweep_id}")
        log.info(f"Using device {self.device}")
        wandb.agent(sweep_id, function=self._run, count=1)

    def _run(self) -> None:
        run = wandb.init()

        hyperparameters = {
            "n_mels": wandb.config.n_mels,
        }

        train_ds = SoundDS(self.train_df, self.device)
        val_ds = SoundDS(self.val_df, self.device)

        log.debug(f"Train DS length: {len(train_ds)}")
        log.debug(f"Val DS length: {len(val_ds)}")

        # num_workers=4 prefetch_factor=2 check out
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        self.train_dl = DataLoader(train_ds, batch_size=wandb.config.batch_size_train, shuffle=True)
        self.val_dl = DataLoader(val_ds, batch_size=wandb.config.batch_size_val, shuffle=False)

        self.model = AudioClassifier(self.class_counts).to(self.device)

        img = wandb.Image(self.val_dl.dataset[0][0][0].cpu().numpy())
        wandb.log({"debug/img": img})

        # Fix: wandb.watch(self.model, log="all") is not working
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
            for data in self.train_dl:
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

                # Normalize the inputs
                # todo: check how to normalize spectrograms
                # log(1 + spectrogram)
                inputs = log(1 + inputs)
                inputs_log: Tensor = torch.log1p(inputs)
                # print(inputs.max())

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

                for name, param in self.model.named_parameters():
                    wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})

                is_first = False

                # Keep stats for Loss and Accuracy
                wandb.log({
                    "train/loss": loss.item(),
                    "debug/time_per_batch": time.time() - timestamp
                })

            self._inference(epoch)
            epoch_duration = time.time() - epoch_time
            log.info(f"Epoch time: {int(epoch_duration // 60)} min {int(epoch_duration % 60)} sec")
        log.info('Finished Training')

    def _inference(self, epoch: int) -> None:
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

                # Normalize the inputs
                # normalize with infos from training set
                # inputs_m: Tensor = inputs.mean()
                # inputs_s: Tensor = inputs.std()
                # inputs: Tensor = (inputs - inputs_m) / inputs_s

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

            wandb.log({
                "val/acc": accuracy_score(y_true, y_pred),
                "val/f1": f1_score(y_true, y_pred, average='weighted')
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
            wandb.log({"val/confusion_matrix": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred)})

        log.info('Finished Inference')
