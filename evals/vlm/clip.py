from typing import Optional, Tuple, Union

import torch
from PIL import Image

import open_clip


class ClipModel:
    def __init__(self, model_name: str, pretrained: str = None, device: str = "cuda"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text: str, images: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")
            images = self.preprocess(images).unsqueeze(0).to(self.device)
        elif isinstance(images, list) and isinstance(images[0], str):
            images = [Image.open(image).convert("RGB") for image in images]
            images = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        text = self.tokenizer(text).to(self.device)

        outputs = self.model(images, text)
        image_features, text_features, logit_scale = outputs
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image
