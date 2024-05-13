import time

import torch
import wandb
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, matthews_corrcoef, cohen_kappa_score, log_loss
from torch import nn
from torch.utils.data import DataLoader

from src.basemodel.dataset import SoundDS
from src.basemodel.classifier import AudioClassifier

torch.cuda.is_available()
if not torch.cuda.is_available():
    raise Exception("GPU not available")


class BasemodelRunner:
    def __init__(self, train_df, val_df):
        self.train_df = train_df
        self.val_df = val_df
        self.train_dl = None
        self.val_dl = None
        self.class_counts: int = self.train_df['species_id'].nunique()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device {self.device}")

        self.model = None

    def run(self):
        sweep_config = {
            "name": "Baseline Sweep",
            "method": "bayes",
            "metric": {"goal": "maximize", "name": "val_acc"},
            "parameters": {
                # "epochs": {"min": 1, "max": 1},
                "epochs": {"values": [1]},
                "learning_rate": {"min": 0, "max": 0.1, "distribution": "log_uniform"},

                "batch_size_train": {"values": [32]},
                "batch_size_val": {"values": [32]},
                "anneal_strategy": {"values": ["linear"]},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project="Baseline-Full", entity="swiss-birder")
        wandb.agent(sweep_id, function=self._run, count=1) # TODO: Change count to 10

    def _run(self):
        run = wandb.init()

        train_ds = SoundDS(self.train_df)
        val_ds = SoundDS(self.val_df)

        # Create training and validation data loaders
        self.train_dl = torch.utils.data.DataLoader(train_ds, batch_size=wandb.config.batch_size_train, shuffle=True)
        self.val_dl = torch.utils.data.DataLoader(val_ds, batch_size=wandb.config.batch_size_val, shuffle=False)

        model = AudioClassifier(self.class_counts + 1)
        self.model = model.to(self.device)

        self._training()

        self._inference()

        model_name = f"./output/{run.id}_{run.name}_model.onnx"
        torch.onnx.export(self.model, torch.randn(1, 2, 64, 64).to(self.device), model_name)

        wandb.save(model_name)
        wandb.finish()

    def _training(self):

        # Loss Function, Optimizer and Scheduler
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=wandb.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=wandb.config.learning_rate,
                                                        steps_per_epoch=int(len(self.train_dl)),
                                                        epochs=wandb.config.epochs,
                                                        anneal_strategy=wandb.config.anneal_strategy)

        # Repeat for each epoch
        for epoch in range(wandb.config.epochs):
            print(f'Epoch: {epoch}')
            epoch_time = time.time()

            # Repeat for each batch in the training set
            for data in self.train_dl:
                timestamp = time.time()
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Keep stats for Loss and Accuracy
                wandb.log({"loss": loss.item()})

                wandb.log({"batch_time": time.time() - timestamp})

            wandb.log({"epoch_time": time.time() - epoch_time})

    def _inference(self):
        # Disable gradient updates
        with torch.no_grad():
            y_pred = []
            y_true = []
            for data in self.val_dl:
                loss_function = nn.CrossEntropyLoss()

                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                y_true.extend(labels.cpu().numpy())

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Get predictions
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)
                wandb.log({"val_loss": loss.item()})

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)
                # Count of predictions that matched the target label
                y_pred.extend(prediction.cpu().numpy())

        wandb.log({"val_acc": accuracy_score(y_true, y_pred)})
        wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=None)})
        wandb.log({"val_f1": f1_score(y_true, y_pred, average='weighted')})
        wandb.log({"val_mcc": matthews_corrcoef(y_true, y_pred)})
        wandb.log({"val_kappa": cohen_kappa_score(y_true, y_pred)})
