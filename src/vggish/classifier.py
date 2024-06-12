from torch import nn

from torchvggish import vggish


class VGGishClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.vggish = vggish()

        # self.vggish = torch.hub.load('harritaylor/torchvggish', 'torchvggish')
        # self.vggish.eval()

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # TODO: Fix shape of input
        x = self.vggish(x)
        x = self.fc(x)
        return x
