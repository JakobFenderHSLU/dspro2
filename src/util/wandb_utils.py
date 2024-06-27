import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


class WandbUtils:
    def __init__(self, wandb_run, class_labels):
        self.wandb_run = wandb_run
        self.class_labels = class_labels

    def log_model_results(self, y_true, y_pred_probabilities, mode="val"):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred_probabilities, axis=1)

        f1 = f1_score(y_true, y_pred, average='weighted')
        self._get_confusion_matrix(y_true, y_pred)

        self.wandb_run.log({
            mode + "/accuracy": accuracy_score(y_true, y_pred),
            mode + "/f1": f1,
            mode + "/roc": wandb.plot.roc_curve(y_true, y_pred_probabilities, self.class_labels, split_table=True),
            mode + "/pr": wandb.plot.pr_curve(y_true, y_pred_probabilities, self.class_labels, split_table=True),
            mode + "/confusion_matrix": wandb.Image("tmp/confusion_matrix.png")
        })

        return f1

    def _get_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        longest_label = max(self.class_labels, key=len)
        class_labels_fixed_length = [label.rjust(len(longest_label)) for label in self.class_labels]

        # Customize plot
        plt.title('Confusion Matrix')
        plt.figure(figsize=(24, 18))
        ax = sns.heatmap(cm, annot=False, cmap='viridis', cbar=False)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_xticks(np.arange(len(self.class_labels)) + 0.5)
        ax.set_yticks(np.arange(len(self.class_labels)) + 0.5)
        ax.set_xticklabels(class_labels_fixed_length, rotation=45)
        ax.set_yticklabels(class_labels_fixed_length, rotation=0)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')

        # Save plot to a file
        plt.savefig("tmp/confusion_matrix.png")
        plt.close()
