"""Inference: load a trained checkpoint and ground a command to one candidate box.

This is the entry point the conversational system will call. It reconstructs the exact backbone,
crop margin, and signal set the checkpoint was trained with, so the feature order always matches.
"""

import numpy as np
import torch

from . import features, signals
from .model import GroundingMLP
from .train import Config, _assemble


class GroundingPredictor:
    def __init__(self, checkpoint: str, device: str | None = None):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = Config(**ckpt["config"])
        self.device = device or features.device()
        self.model = GroundingMLP(ckpt["input_dim"], self.cfg.hidden_dim, self.cfg.dropout)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.val_ap50 = ckpt["val_ap50"]

        model_id, self.dim = features.BACKBONES[self.cfg.backbone]
        self._clip = None
        self._model_id = model_id

    @property
    def clip(self):
        if self._clip is None:
            from sentence_transformers import SentenceTransformer
            self._clip = SentenceTransformer(self._model_id, device=self.device)
        return self._clip

    def score(self, image, proposals, proposal_classes, proposal_scores, command):
        """Return (scores over candidates, sample dict). Higher score = better match."""
        proposals = np.asarray(proposals, dtype=np.float32)
        crops = [features.crop_box(image, b, self.cfg.crop_margin) for b in proposals]
        img_feat = self.clip.encode(crops, convert_to_numpy=True, device=self.device,
                                    batch_size=64, show_progress_bar=False)
        img_feat = (img_feat / (np.linalg.norm(img_feat, axis=-1, keepdims=True) + 1e-8))[None].astype(np.float16)
        txt = self.clip.encode(command, convert_to_numpy=True, device=self.device, show_progress_bar=False)
        txt = (txt / (np.linalg.norm(txt) + 1e-8))[None].astype(np.float16)

        sample = {"command": command, "proposals": proposals,
                  "proposal_classes": list(proposal_classes),
                  "proposal_scores": np.asarray(proposal_scores, dtype=np.float32)}
        blocks = [signals.build_signal_block(sample, self.cfg.signals, image)] if self.cfg.signals else None

        feats = _assemble(img_feat, txt, blocks)[0]
        with torch.no_grad():
            scores = self.model(torch.from_numpy(feats)).numpy()
        return scores, sample

    def predict(self, image, proposals, proposal_classes, proposal_scores, command) -> int:
        """Return the index of the candidate the command refers to."""
        scores, _ = self.score(image, proposals, proposal_classes, proposal_scores, command)
        return int(scores.argmax())
