"""Frozen image backbones used as feature extractors for the dish classifier."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / ".hf_cache")

# name: (Hugging Face id, pinned revision)
BACKBONES = {
    "clip-b16": ("openai/clip-vit-base-patch16", "57c216476eefef5ab752ec549e440a49ae4ae5f3"),
    "dinov2-b": ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
    "siglip-b16": ("google/siglip-base-patch16-224", "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"),
    "clip-l14": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
}


class Backbone:
    def __init__(self, name: str, local_dir: str | Path | None = None):
        """local_dir: a fine-tuned SigLIP vision tower saved with save_pretrained (see kaggle/train_plate.py).
        If it also holds preprocessor_config.json (plate/export_processor.py), nothing is downloaded."""
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, SiglipModel, SiglipVisionModel

        self.name = name
        mid, rev = BACKBONES[name]
        if local_dir is not None and (Path(local_dir) / "preprocessor_config.json").exists():
            self.processor = AutoImageProcessor.from_pretrained(str(local_dir))
        else:
            self.processor = AutoImageProcessor.from_pretrained(mid, revision=rev, cache_dir=CACHE)
        if local_dir is not None:
            self.model = SiglipVisionModel.from_pretrained(str(local_dir)).eval()
            self.kind = "siglip_vision"
        elif name.startswith("clip"):
            self.model = CLIPModel.from_pretrained(mid, revision=rev, cache_dir=CACHE).eval()
            self.kind = "clip"
        elif name.startswith("siglip"):
            self.model = SiglipModel.from_pretrained(mid, revision=rev, cache_dir=CACHE).eval()
            self.kind = "clip"
        else:
            self.model = AutoModel.from_pretrained(mid, revision=rev, cache_dir=CACHE).eval()
            self.kind = "dino"

    @torch.inference_mode()
    def embed(self, images: list[Image.Image], batch: int = 32) -> np.ndarray:
        """L2-normalised embeddings, one row per image."""
        out = []
        for s in range(0, len(images), batch):
            x = self.processor(images=images[s:s + batch], return_tensors="pt")
            if self.kind == "siglip_vision":
                e = self.model(pixel_values=x["pixel_values"]).pooler_output  # = SiglipModel.get_image_features
            elif self.kind == "clip":
                e = self.model.get_image_features(pixel_values=x["pixel_values"])
                e = getattr(e, "pooler_output", e)
            else:
                h = self.model(pixel_values=x["pixel_values"]).last_hidden_state
                e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)  # CLS + mean patch (DINOv2 linear-probe recipe)
            e = e.float()
            out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)
