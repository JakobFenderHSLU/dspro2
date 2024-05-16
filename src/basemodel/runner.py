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

log = init_logging("basemodel", level=logging.INFO)


class BasemodelRunner:
    def __init__(self, train_df, val_df) -> None:
        self.train_df: pd.DataFrame = train_df
        self.val_df: pd.DataFrame = val_df
        self.train_dl: DataLoader = None
        self.val_dl: DataLoader = None
        self.class_counts: int = len(train_df.columns) - 1
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: AudioClassifier = None

        self.loss_function: CrossEntropyLoss = nn.CrossEntropyLoss()

    def run(self) -> None:
        sweep_config: dict = {
            "name": "Baseline Sweep",
            "method": "grid",
            "metric": {"goal": "maximize", "name": "val_f1"},
            "parameters": {
                "epochs": {"value": 20},
                "learning_rate": {"value": 0.1},  # {"min": 0.0001, "max": 0.1},
                "batch_size_train": {"values": [64]},  # try higher
                "batch_size_val": {"values": [64]},  # try higher
                "anneal_strategy": {"values": ["linear"]},
            }
        }
        sweep_id: str = wandb.sweep(sweep=sweep_config, project="Baseline-Full", entity="swiss-birder")
        wandb.agent(sweep_id, function=self._run, count=50)

    def _run(self) -> None:
        run = wandb.init()
        log.info(f"Using device {self.device}")

        train_ds = SoundDS(self.train_df, self.device)
        val_ds = SoundDS(self.val_df, self.device)

        self.train_dl = DataLoader(train_ds, batch_size=wandb.config.batch_size_train,
                                   shuffle=True, num_workers=4, prefetch_factor=2)  # num_workers=4 prefetch_factor=2
        self.val_dl = DataLoader(val_ds, batch_size=wandb.config.batch_size_val,
                                 shuffle=False)  # num_workers=16 prefetch_factor=2
        self.model = AudioClassifier(self.class_counts).to(self.device)

        img = wandb.Image(self.val_dl.dataset[0][0][0].cpu().numpy())
        wandb.log({"val/img": img})

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

        # Repeat for each epoch
        for epoch_iter in range(wandb.config.epochs):
            epoch = epoch_iter + 1
            log.info(f'Epoch: {epoch}')
            wandb.log({"epoch": epoch})
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

                # Normalize the inputs
                # todo: check how to normalize spectrograms
                # log(1 + spectrogram)
                # inputs = log(1 + inputs)
                # inputs_log: Tensor = torch.log1p(inputs)
                # print(inputs.max())

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                preds: Tensor = self.model(inputs)
                loss: Tensor = self.loss_function(preds, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Keep stats for Loss and Accuracy
                wandb.log({"train/loss": loss.item()})

                wandb.log({"debug/time_per_batch": time.time() - timestamp})

            torch.cuda.synchronize()

            epoch_duration = time.time() - epoch_time
            wandb.log({"epoch_duration": epoch_duration})

            self._inference()

            log.info(f"Epoch time: {int(epoch_duration // 60)} min {int(epoch_duration % 60)} sec")
        log.info('Finished Training')

    def _inference(self):
        log.info("Starting Inference")
        # Disable gradient updates
        with (torch.no_grad()):
            y_pred = []
            y_true = []
            for data in self.val_dl:
                # Get the input features and target labels, and put them on the GPU
                inputs: Tensor = data[0].to(self.device)
                labels: Tensor = data[1].to(self.device)

                y_true.extend(labels.cpu().numpy())

                # Normalize the inputs
                # normalize with infos from training set
                # inputs_m: Tensor = inputs.mean()
                # inputs_s: Tensor = inputs.std()
                # inputs: Tensor = (inputs - inputs_m) / inputs_s

                # Get predictions
                preds: Tensor = self.model(inputs)
                loss: Tensor = self.loss_function(preds, labels)
                wandb.log({"val/loss": loss.item()})

                # Count of predictions that matched the target label
                y_pred.extend(preds.cpu().numpy())

            # y_pred make the highest value 1 and the rest 0 for this np array
            y_pred_argmax = np.argmax(y_pred, axis=1)
            new_y_pred = np.zeros_like(y_pred)
            new_y_pred[np.arange(len(y_pred)), y_pred_argmax] = 1

            y_pred = new_y_pred

            wandb.log({"val/acc": accuracy_score(y_true, y_pred)})
            wandb.log({"val/f1": f1_score(y_true, y_pred, average='weighted')})
        log.info('Finished Inference')
