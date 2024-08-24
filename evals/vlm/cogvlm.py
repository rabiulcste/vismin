from collections import defaultdict
from typing import Union

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer

from ..dataset_zoo.config import DATASET_INFO, get_all_splits
from ..metrics.itm import (compute_mllm_score_itg, compute_mllm_score_standard,
                           is_match_with_label)
from ..utils import get_combined_image


class CogVLMModel:
    def __init__(self, model_name="THUDM/cogvlm-chat-hf", device="cuda"):
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
            .to("cuda")
            .eval()
        )
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.device = device

    def predict(self, text: str, image: Union[Image.Image, list]):
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
        # Note: We put two images side-by-side to form one as the model expects a single image
        elif isinstance(image, list):
            image = [Image.open(img).convert("RGB") for img in image]
            image = get_combined_image(image)

        inputs = self.model.build_conversation_input_ids(
            self.tokenizer, query=text, images=[image], template_version="vqa"
        )
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to("cuda"),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to("cuda"),
            "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0].strip()

        return response

    def predict_batch(self, texts, images, max_new_tokens=128):
        return [self.predict(text, image) for text, image in zip(texts, images)]
