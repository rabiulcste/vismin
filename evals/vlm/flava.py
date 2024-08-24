import PIL
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, FlavaForPreTraining

from ..dataset_zoo.config import DATASET_INFO, get_all_splits
from ..metrics.itm import compute_contrastive_score_itg


class FlavaModel:
    def __init__(self, model_name="facebook/flava-full", device="cuda", **kwargs):
        self.model = FlavaForPreTraining.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def predict(self, text, images):
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")
        elif isinstance(images, list) and isinstance(images[0], str):
            images = [Image.open(img).convert("RGB") for img in images]
        else:
            images = images

        extended_axis = None
        if isinstance(text, str) and isinstance(images, list):
            text = [text] * len(images)
            extended_axis = 1
        elif isinstance(text, list) and isinstance(images, PIL.Image.Image):
            images = [images] * len(text)
            extended_axis = 0

        assert len(text) == len(images)

        inputs = self.process_input(
            text=text,
            image=images,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.contrastive_logits_per_image

        # Return only the logits for the exact number of inputs
        if extended_axis == 1:
            return logits_per_image[:, :1]
        elif extended_axis == 0:
            return logits_per_image[:1]
        else:
            return logits_per_image

    def predict_itm(self, texts: list, images: list):
        inputs_c0_i0 = self.process_input(texts[0], images[0])
        inputs_c1_i0 = self.process_input(texts[1], images[0])
        inputs_c0_i1 = self.process_input(texts[0], images[1])
        inputs_c1_i1 = self.process_input(texts[1], images[1])

        outputs_c0_i0 = self.model(**inputs_c0_i0)
        outputs_c1_i0 = self.model(**inputs_c1_i0)
        outputs_c0_i1 = self.model(**inputs_c0_i1)
        outputs_c1_i1 = self.model(**inputs_c1_i1)

        contrastive_scores = self.extract_contrastive_scores(outputs_c0_i0, outputs_c1_i0, outputs_c0_i1, outputs_c1_i1)
        itm_scores = self.extract_itm_scores(outputs_c0_i0, outputs_c1_i0, outputs_c0_i1, outputs_c1_i1)

        return contrastive_scores, itm_scores

    def process_input(self, text, image):
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            return_codebook_pixels=True,
            return_image_mask=True,
        ).to(self.device)

        inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
        inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])

        return inputs

    def extract_contrastive_scores(self, *outputs):
        return [output.contrastive_logits_per_image.item() for output in outputs]

    def extract_itm_scores(self, *outputs):
        return [torch.nn.functional.softmax(output.itm_logits)[0][1].item() for output in outputs]
