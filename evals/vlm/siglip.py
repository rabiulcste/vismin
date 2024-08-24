from typing import Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SiglipModel:
    def __init__(self, model_name: str = "google/siglip-base-patch16-256", device: Optional[str] = "cuda", **kwargs):
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def predict(self, text: str, images: Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")
        elif isinstance(images, list) and isinstance(images[0], str):
            images = [Image.open(img).convert("RGB") for img in images]

        inputs = self.processor(
            text=text, images=images, padding="max_length", return_tensors="pt", truncation=True
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image
