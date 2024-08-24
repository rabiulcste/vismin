import torch
from PIL import Image

from .model_zoo.blip_models import BLIPModelWrapper


class BlipModel:
    def __init__(self, model_name="blip-coco-base", device="cuda", **kwargs):
        self.model = BLIPModelWrapper(model_name, device)

    def predict(self, text, images):
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")
        if isinstance(images, list) and isinstance(images[0], str):
            images = [Image.open(img).convert("RGB") for img in images]

        text_embedding, _, _ = self.model.get_text_embeddings(text)
        image_embedding = self.model.get_image_embeddings(images)
        logits_per_image = torch.matmul(image_embedding, text_embedding.T)
        print(logits_per_image)
        return logits_per_image

    def predict_itm(self, texts: list, images: list):
        text_embedding_0, _, _ = self.model.get_text_embeddings(texts[0])
        text_embedding_1, _, _ = self.model.get_text_embeddings(texts[1])
        image_embedding_0 = self.model.get_image_embeddings(images[0])
        image_embedding_1 = self.model.get_image_embeddings(images[1])

        image_embedding = torch.cat([image_embedding_0, image_embedding_1], dim=0)
        text_embedding = torch.cat([text_embedding_0, text_embedding_1], dim=0)
        logits_per_image = torch.matmul(image_embedding, text_embedding.T)

        return logits_per_image
