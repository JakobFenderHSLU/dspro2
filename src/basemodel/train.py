import pandas as pd
import torch
import wandb
import util

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from importlib import reload
from util import SoundDS, AudioClassifier, training, inference
import os
import pandas as pd

torch.cuda.is_available()
if not torch.cuda.is_available():
    raise Exception("GPU not available")

# get all folder names in directory


def train():
    wandb.init()

    myds = SoundDS(df, "")

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=wandb.config.batch_size_train,
                                           shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds,
                                         batch_size=wandb.config.batch_size_val,
                                         shuffle=False)

    # Create the model and put it on the GPU if available
    model = AudioClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    model = model.to(device)

    num_epochs = wandb.config.epochs
    training(model, train_dl, num_epochs, device)

    # Run inference on trained model with the validation set
    inference(model, val_dl, device)

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, torch.randn(1, 1, 128, 201), "model.onnx")

    wandb.save("model.onnx")


sweep_config = {
    "name": "Baseline Sweep",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "epochs": {"min": 7, "max": 20},
        "learning_rate": {"min": 0, "max": 0.1, "distribution": "log_uniform"},

        "batch_size_train": {"values": [32]},
        "batch_size_val": {"values": [32]},
        "anneal_strategy": {"values": ["linear"]},
    }
}
sweep_id = wandb.sweep(sweep=sweep_config, project="Baseline-Full", entity="swiss-birder")
wandb.agent(sweep_id, function=train)
