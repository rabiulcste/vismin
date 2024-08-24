from typing import Union

import PIL
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from ..utils import get_combined_image


class LlaVaModel:
    def __init__(self, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf", device: str = "cuda", load_in_Nbit=None):
        self.model_name_or_path = model_name_or_path
        gpu_name = torch.cuda.get_device_name()
        if "A100" in gpu_name or "H100" in gpu_name:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = None

        processor = LlavaNextProcessor.from_pretrained(model_name_or_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_Nbit == 4,
            load_in_8bit=load_in_Nbit == 8,
            attn_implementation=attn_implementation,
        )
        if load_in_Nbit is None:  # we are not using 4-bit model, setting device to GPU manually
            model = model.to(device)

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

        self.processor = processor
        self.model = model
        self.device = device

    def prepare_input(selt, text):
        prompt = f"[INST] {text} [/INST]"
        prompt = "<image> " + prompt
        return prompt

    def predict(self, text: str, image: Union[PIL.Image.Image, list], max_new_tokens=64):
        if isinstance(image, str):
            image = PIL.Image.open(image).convert("RGB")
        elif isinstance(image, list):
            if isinstance(image[0], str):
                image = [PIL.Image.open(img_path).convert("RGB") for img_path in image]
            image = get_combined_image(image)
        text = self.prepare_input(text)

        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device, dtype=torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        output = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return output

    def predict_batch_legacy(self, texts: list, images: list, max_new_tokens=20):
        texts = [f"[INST] {text} [/INST]" for text in texts]
        if isinstance(images[0], list):
            images = [get_combined_image(image_list) for image_list in images]

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(
            self.device, dtype=torch.float16
        )
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs

    def predict_batch(self, texts: list, images: list, max_new_tokens=20):
        return [self.predict(text, image_pair) for text, image_pair in zip(texts, images)]
