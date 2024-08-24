from typing import Union

import PIL
import torch
from transformers import (AutoModelForVision2Seq, AutoProcessor,
                          BitsAndBytesConfig)


class Idefics2Model:
    def __init__(self, model_name_or_path="HuggingFaceM4/idefics2-8b", device: str = "cuda", **kwargs):
        gpu_name = torch.cuda.get_device_name()
        if "A100" in gpu_name or "H100" in gpu_name:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = None

        load_in_Nbit = kwargs.pop("load_in_Nbit", None)
        if model_name_or_path == "HuggingFaceM4/idefics2-8b" and load_in_Nbit == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                # bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            quantization_config = None
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
            _attn_implementation=attn_implementation,  # only A100, H100 GPUs
            # if model_name_or_path in ["HuggingFaceM4/idefics2-8b", "HuggingFaceM4/idefics2-8b-base"] else None,
            quantization_config=quantization_config,
        )
        self.processor = processor
        self.model = model
        self.device = device

    def prepare_prompt(self, text, image):
        if isinstance(image, list):
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                },
            ]
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        message = self.processor.apply_chat_template(message, add_generation_prompt=True)
        return message

    def predict(self, text: str, image: Union[PIL.Image.Image, list], max_new_tokens=64):
        if isinstance(image, list):
            if isinstance(image[0], str):
                image = [PIL.Image.open(img).convert("RGB") for img in image]
        elif isinstance(image, str):
            image = [PIL.Image.open(image).convert("RGB")]

        prompt = self.prepare_prompt(text, image)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, dtype=torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2)
        if self.processor.tokenizer.padding_side == "left":
            generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        output = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return output

    def predict_batch(self, texts, images, max_new_tokens=20):
        if isinstance(images, list):
            pil_images = []
            for image in images:
                if isinstance(image, str):
                    image = PIL.Image.open(image).convert("RGB")
                    pil_images.append([image])
                elif isinstance(image, list) and isinstance(image[0], str):
                    pil_images.append([PIL.Image.open(img).convert("RGB") for img in image])

        # turn the images into a list of list, each containing a single image
        prompts = [self.prepare_prompt(text, image) for text, image in zip(texts, images)]
        inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(
            self.device, dtype=torch.float16
        )
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        if self.processor.tokenizer.padding_side == "left":
            generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs
