import pandas as pd
import torch
from src.embedder.BEATs import BEATs, BEATsConfig
from src.util.logger_utils import init_logging

log = init_logging("bead embedder")


class BeatEmbedder:
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, scale: str = "debug",
                 model_path: str = './input/models/BEATs_iter3_plus_AS2M.pt') -> None:
        self.train_df: pd.DataFrame = train_df
        self.val_df: pd.DataFrame = val_df
        self.scale: str = scale
        self.model_path: str = model_path

        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self) -> None:
        self._run()

    def _run(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        cfg = BEATsConfig(checkpoint['cfg'])
        self.beats_model = BEATs(cfg)
        self.beats_model.load_state_dict(checkpoint['model'])
        self.beats_model.eval()

        # extract the audio representation
        audio_input_16khz = torch.randn(1, 10000)

        representation = self.beats_model.extract_features(audio_input_16khz)[0]
        log.info(f"Audio representation shape: {representation.shape}")
        log.info(representation)