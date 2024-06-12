import torch

from src.util.logger_utils import init_logging
from src_external.BEATs import BEATs, BEATsConfig

log = init_logging("bead feature_extractor")


class BeatEmbedder:
    def __init__(self, model_path: str = './input/models/BEATs_iter3_plus_AS2M.pt',
                 device: torch.device = None) -> None:
        self.model_path: str = model_path
        self.device: torch.device = device

        checkpoint = torch.load(self.model_path, map_location=self.device)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.beats_model = BEATs(cfg)
        self.beats_model.load_state_dict(checkpoint['model'])
        self.beats_model.eval()

    def embed(self, audio) -> torch.Tensor:
        signal, sr = audio

        assert sr == 16_000, f"Expected sample rate of 16_000, got {sr}"

        return self.beats_model.extract_features(signal)[0]
