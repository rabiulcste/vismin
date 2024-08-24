import os
import re
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers import AutoPipelineForInpainting
from groundingdino.util import box_ops
from PIL import Image
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              build_sam, sam_model_registry)
from word2number import w2n

from commons.constants import CHECKPOINT_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"



class SamHelper:
    def __init__(self, gen_type="automatic"):
        sam_checkpoint = os.path.join(CHECKPOINT_DIR, "sam_vit_h_4b8939.pth")

        if not os.path.exists(sam_checkpoint):
            download_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            log_message = (
                f"Checkpoint file not found: {sam_checkpoint}\n"
                f"Please download the required checkpoint file using the following command:\n"
                f"wget -c {download_url} -P {CHECKPOINT_DIR}\n"
                f"After downloading, ensure the file is located at: {sam_checkpoint}"
            )
            raise FileNotFoundError(log_message)
        if gen_type == "automatic":
            self.mask_generator = self.load_sam_generator(sam_checkpoint)
        else:
            self.sam_predictor = self.load_sam_model(sam_checkpoint)

    @staticmethod
    def load_sam_generator(sam_checkpoint):
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        print("SAM model loaded from {}".format(sam_checkpoint))
        return mask_generator

    @staticmethod
    def load_sam_predictor(sam_checkpoint):
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        print("SAM model loaded from {}".format(sam_checkpoint))

        return sam_predictor

    def mask_predictor(self, image, boxes):
        self.sam_predictor.set_image(image)  # set image
        H, W, _ = image.shape  # box: normalized box xywh -> unnormalized xyxy
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(device)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        print(f"no of sam masks: {len(masks)}")
        return masks

    def automatic_mask_generator(self, image):
        masks = self.mask_generator.generate(image)
        return masks

    def process_image(self, image_source, boxes):
        sam_masks = self.generate_sam_mask(image_source, boxes)
        if sam_masks.numel() != 0:
            combined_mask = torch.zeros_like(sam_masks[0][0], dtype=torch.bool)
            for mask in sam_masks:
                combined_mask = torch.logical_or(combined_mask, mask[0])
            image_mask = combined_mask.cpu().numpy()

        image_source = Image.fromarray(image_source)
        image_mask = Image.fromarray(image_mask)

        return image_source, image_mask


class BackgroundDescriptor:
    def __init__(self, model_name_or_path="Salesforce/blip2-flan-t5-xxl"):
        print(f"Initalizing {self.__class__.__name__} class with model: {model_name_or_path}")
        self.captioning_model, self.processor = self.load_pretrained_processor_model(model_name_or_path)

    @staticmethod
    def load_pretrained_processor_model(model_name_or_path):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        processor = Blip2Processor.from_pretrained(model_name_or_path)
        if "opt" in model_name_or_path:
            processor.tokenizer.padding_side = "left"

        if "t5" in model_name_or_path and torch.cuda.device_count() > 1:
            device_map = {
                "query_tokens": 0,
                "vision_model": 0,
                "language_model": 1,
                "language_projection": 0,
                "qformer": 0,
            }
        else:
            device_map = "auto"

        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, load_in_4bit=True, device_map=device_map, torch_dtype=torch.float16, load_in_8bit=True
        )
        return model, processor

    def get_background_description(self, images, object_names: Union[str, List[str]]):
        if isinstance(images, list) and isinstance(images[0], np.ndarray):
            images = [Image.fromarray(image) for image in images]
        prompts = [
            "Describe only the background of this image. Do not describe the objects in the image. Background is"
        ]
        images = [image.convert("RGB") for image in images]

        inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(
            device, dtype=torch.float16
        )
        generated_ids = self.captioning_model.generate(
            **inputs,
            max_new_tokens=10,
            length_penalty=-1,
            num_return_sequences=5,
            do_sample=True,
            temperature=0.5,
            top_k=40,
        )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        # print(f"generated text: {generated_texts}")
        return self.handle_text(generated_texts, object_names)

    @staticmethod
    def handle_text(generated_texts: list, object_names):
        # If obj_name is a string, convert it to a list
        if isinstance(object_names, str):
            object_names = [object_names]

        for text in generated_texts:
            if not any(name in text for name in object_names):
                return text
        return "a photo, outdoor"


class SDXLInpainting:
    def __init__(
        self,
        device="cuda",
    ):
        repo_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        print(f"Initializing {self.__class__.__name__} class with model: {repo_id} and device: {device}")
        self.device = device
        self.load_model()

    def load_model(self):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(device)

    def generate_inpainting(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        num_images_per_prompt: int = None,
        strength: float = None,
    ):
        image_inpainting = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            strength=strength,
            output_type="pil",
        ).images

        return image_inpainting


class ObjectCountParser:
    DIGIT_TO_TEXT = {
        "0": "no",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
    }

    def __init__(self):
        import spacy
        from nltk.stem import WordNetLemmatizer

        print(f"Initalizing {self.__class__.__name__} class")
        self._language_model = spacy.load("en_core_web_sm")
        self._lemmatizer = WordNetLemmatizer()

    def to_singular_spacy(self, phrase):
        doc = self._language_model(phrase)

        # Reconstruct the phrase with singular forms
        singular_phrase = ""
        for token in doc:
            # Check if the token is a noun (plural) to convert it to singular
            if token.pos_ == "NOUN" and token.tag_ == "NNS":
                singular_phrase += token.lemma_ + " "
            else:
                singular_phrase += token.text + " "

        return singular_phrase.strip()

    def _replace_a_with_one(self, text, verbose=False):
        # Use a regular expression to replace 'a' or 'A' with 'one' when it stands alone as a word
        # The pattern looks for 'a' or 'A' surrounded by word boundaries (\b), making the replacement case-insensitive and more flexible
        output = re.sub(r"\b([aA]|[aA]n)\b", "one", text)
        if output != text and verbose:
            print(f"Replaced 'a' with 'one' in the caption: {text} => {output}")
        return output

    def _parse_objects_from_caption(self, caption_text, apply_a_to_one=False):
        """Parse objects from caption."""
        if apply_a_to_one:
            caption_text_updated = self._replace_a_with_one(caption_text)
        else:
            caption_text_updated = caption_text
        parsed_caption = self._language_model(caption_text_updated)
        noun_chunks = [chunk.text for chunk in parsed_caption.noun_chunks]
        return noun_chunks, caption_text_updated

    def _extract_count_from_chunk(self, chunk):
        """Extract count from chunk."""
        words = chunk.split()

        # Check if first word can be converted to a number
        try:
            count_t = words[0]
            count_n = w2n.word_to_num(words[0].lower())
            object_name = " ".join(words[1:])
            object_name = self.to_singular_spacy(object_name)
        except ValueError:
            count_n = None
            count_t = None
            object_name = chunk

        return object_name, count_n, count_t


def get_obj_removal_verify_qa_pairs_by_name(object_name):
    return [
        {
            "question": f"Do you see a {object_name} in the image?",
            "choices": ["yes", "no"],
            "answer": "no",
        },
        {
            "question": f"Is there a {object_name} present in the image?",
            "choices": ["present", "not present"],
            "answer": "not present",
        },
        {
            "question": f"In the image, is there a {object_name}?",
            "choices": ["yes", "no"],
            "answer": "no",
        },
    ]


def get_updated_count_phrase(self, obj_name: str, obj_cnt_n: int, obj_cnt_t: str):
    new_count_n = obj_cnt_n - 1
    new_count_t = self.obj_cnt_parser.DIGIT_TO_TEXT.get(str(new_count_n))
    original_count_phrase = f"{obj_cnt_t} {obj_name}"
    if new_count_n == 1:
        obj_name = self.obj_cnt_parser.to_singular_spacy(obj_name)
    updated_count_phrase = f"{new_count_t} {obj_name}"

    return new_count_n, new_count_t, original_count_phrase, updated_count_phrase


# TODO: replace rule based phrase replacement with language model based approach
def get_new_caption_for_updated_count(caption_text, original_count_phrase, updated_count_phrase):
    new_caption_text = None
    try:
        start_index = caption_text.lower().find(original_count_phrase.lower())
        end_index = start_index + len(original_count_phrase)
        new_caption_text = caption_text[:start_index] + updated_count_phrase + caption_text[end_index:]
    except Exception as e:
        print(f"Error: {e}")

    return new_caption_text


def calculate_iou(box1, box2):
    # Unpack coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersections
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Calculate intersection width and height
    iw = max(xi2 - xi1, 0)
    ih = max(yi2 - yi1, 0)

    # Calculate intersection and union areas
    intersection_area = iw * ih
    union_area = w1 * h1 + w2 * h2 - intersection_area

    # Compute IoU
    if union_area > 0:
        return intersection_area / union_area
    else:
        return 0


def filter_boxes_optimized(bboxes1, bboxes2, iou_threshold=0.2):
    # Sort both lists by the x coordinate
    bboxes1_sorted = sorted(bboxes1, key=lambda x: x[0])
    bboxes2_sorted = sorted(bboxes2, key=lambda x: x[0])

    filtered_boxes = []
    j_start = 0
    n = len(bboxes2_sorted)

    for box1 in bboxes1_sorted:
        keep_box = True
        # Start from the closest x in bboxes2
        for j in range(j_start, n):
            box2 = bboxes2_sorted[j]
            # Stop if further boxes are too far right to overlap
            if box2[0] > box1[0] + box1[2]:
                break
            if box1[0] > box2[0] + box2[2]:
                j_start = j + 1
                continue
            # Calculate IoU only if the boxes might overlap
            iou = calculate_iou(box1, box2)
            if iou > iou_threshold:
                keep_box = False
                break
        if keep_box:
            filtered_boxes.append(box1)

    return filtered_boxes
