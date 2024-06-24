import numpy as np
import wandb
from sklearn.metrics import accuracy_score, f1_score


class WandbUtils:
    def __init__(self, wandb_run, class_labels):
        self.wandb_run = wandb_run
        self.class_labels = class_labels

    def log_model_results(self, y_true, y_pred_probabilities, mode="val"):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred_probabilities, axis=1)

        f1 = f1_score(y_true, y_pred, average='weighted')

        self.wandb_run.log({
            mode + "/accuracy": accuracy_score(y_true, y_pred),
            mode + "/f1": f1,
            mode + "/roc": wandb.plot.roc_curve(y_true, y_pred_probabilities, self.class_labels, split_table=True),
            mode + "/pr": wandb.plot.pr_curve(y_true, y_pred_probabilities, self.class_labels, split_table=True),
            mode + "/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=self.class_labels,
                split_table=True)
        })

        return f1
