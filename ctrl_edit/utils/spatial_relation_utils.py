import re

import cv2
import numpy as np
from PIL import Image

from commons.constants import LAMA_CKPT, LAMA_CONFIG, VALID_SPATIAL_DIRECTIONS

from ..lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from .helpers import get_xyxy_from_xywh, pluralize


def adjust_boxes_touching_edges(boxes, image_width, image_height, shift_amount=5):
    """
    Adjusts boxes that are touching the edges of the image, shifting them slightly towards the center.

    Args:
    - boxes: List of boxes in the format [(name, (x, y, w, h)), ...] or [{'name': name, 'bounding_box': (x, y, w, h)}, ...]
    - image_width: Width of the image
    - image_height: Height of the image
    - shift_amount: Amount to shift the boxes towards the center (in pixels)

    Returns:
    - adjusted_boxes: List of adjusted boxes in the same format as the input
    """
    adjusted_boxes = []
    for box in boxes:
        if isinstance(box, dict):
            box_dict_format = True
            name, (bbox_x, bbox_y, bbox_w, bbox_h) = box["name"], box["bounding_box"]
        else:
            box_dict_format = False
            name, (bbox_x, bbox_y, bbox_w, bbox_h) = box

        # Adjust if touching left or right edge
        if bbox_x <= 0:
            bbox_x += shift_amount
        elif bbox_x + bbox_w >= image_width:
            bbox_x -= shift_amount

        # Adjust if touching top or bottom edge
        if bbox_y <= 0:
            bbox_y += shift_amount
        elif bbox_y + bbox_h >= image_height:
            bbox_y -= shift_amount

        # Reconstruct the box with the adjusted coordinates
        if box_dict_format:
            adjusted_box = {"name": name, "bounding_box": (bbox_x, bbox_y, bbox_w, bbox_h)}
        else:
            adjusted_box = (name, (bbox_x, bbox_y, bbox_w, bbox_h))

        adjusted_boxes.append(adjusted_box)

    return adjusted_boxes


def blank_out_image_regions_by_bounding_boxes(image: Image.Image, bounding_boxes: list):
    """
    Blank out a region from the image given the bounding box.

    Args:
    - image: PIL Image object
    - bbox: Bounding box in the format [x, y, w, h]

    Returns:
    - image: PIL Image object with the region blanked out
    """
    image_np = np.array(image)
    for bbox in bounding_boxes:
        x0, y0, x1, y1 = get_xyxy_from_xywh(bbox)
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        image_np[y0:y1, x0:x1] = 0
    return Image.fromarray(image_np)


def normalize_bounding_box(box, height, width):
    # box: x, y, w, h (in 512 format) -> x_min, y_min, x_max, y_max
    x_min, y_min = box[0] / width, box[1] / height
    w_box, h_box = box[2] / width, box[3] / height

    x_max, y_max = x_min + w_box, y_min + h_box

    return x_min, y_min, x_max, y_max


def create_mask_image_from_bbox(bboxes: list, image_height: int, image_width: int):
    bbox_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)
        bbox_mask[y_min : y_min + height, x_min : x_min + width] = 255

    return bbox_mask


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1)
    return mask


def load_llama_model_for_inpainting(device="cuda"):
    model = build_lama_model(LAMA_CONFIG, LAMA_CKPT, device=device)
    return model, inpaint_img_with_builded_lama


def remove_object_from_image_using_llama_model(
    model, input_image: Image.Image, bounding_boxes: list, inpaint_img_with_builded_lama, dilate_kernel_size=None
):
    image_width, image_height = input_image.size
    if not isinstance(input_image, np.ndarray):
        inp_image = np.array(input_image)
    else:
        inp_image = input_image

    inp_mask = create_mask_image_from_bbox(bounding_boxes, image_height, image_width)

    if dilate_kernel_size is not None:
        inp_mask = dilate_mask(inp_mask, dilate_kernel_size)
    inp_img_cleaned = inpaint_img_with_builded_lama(
        model,
        inp_image,
        inp_mask,
    )
    inp_img_cleaned = Image.fromarray(inp_img_cleaned)
    return inp_img_cleaned


def swap_phrases_in_spatial_prompt(prompt, bounding_box_data):
    """
    Swap phrases in the given prompt based on their descriptions in the bounding_boxes dictionary.

    Parameters:
    - prompt: The original prompt text as a string.
    - bounding_boxes: A dictionary with phrases and their bounding boxes.

    Returns:
    - A new prompt with the phrases swapped.
    """
    prompt = prompt.lower()
    # Extract phrases from the bounding_boxes dictionary
    phrases = [item[0] for item in bounding_box_data]

    # Make sure there are exactly two phrases to swap
    if len(phrases) != 2:
        return None

    # Find the positions of the phrases in the prompt
    match1 = re.search(phrases[0], prompt)
    match2 = re.search(phrases[1], prompt)

    # Only proceed if both phrases were found
    if match1 and match2:
        # Swap the phrases
        new_prompt = (
            prompt[: match1.start()]
            + phrases[1]
            + prompt[match1.end() : match2.start()]
            + phrases[0]
            + prompt[match2.end() :]
        )
        return new_prompt

    return None


def get_obj_existence_verify_qa_pairs_by_name(object_name, choices):
    article_match = re.match(r"^(a|an) (.+)$", object_name)
    article, object_name_wo_article = "", ""
    if article_match:
        # print(f"{object_name} -> {article_match.groups()}")
        article, object_name_wo_article = article_match.groups()

    qa_pairs = [
        {
            "question": "What is in the image?",
            "choices": choices,
            "answer": object_name,
        },
        # {
        #     "question": f"Does the object in the image look significantly deformed or misshapen?",
        #     "choices": ["yes", "no"],
        #     "answer": "no",
        # },
        {
            "question": "What object is visible in the image?",
            "choices": choices,
            "answer": object_name,
        },
        {
            "question": f"Is this {object_name}?",
            "choices": ["yes", "no"],
            "answer": "yes",
        },
        {
            "question": f"Do you see {object_name} in the image?",
            "choices": ["yes", "no"],
            "answer": "yes",
        },
        {
            "question": f"Is there {object_name} present in the image?",
            "choices": ["yes", "no"],
            "answer": "yes",
        },
        {
            "question": f"In the image, is there {object_name}?",
            "choices": ["yes", "no"],
            "answer": "yes",
        },
    ]

    if article:
        plural_object_name = pluralize(object_name_wo_article)
        qa_pairs.extend(
            [
                {
                    "question": f"How many {plural_object_name} are there in the image?",
                    "choices": ["0", "1", "2", "3"],
                    "answer": "1",
                },
                {
                    "question": f"Are there multiple {plural_object_name} in the image?",
                    "choices": ["yes", "no"],
                    "answer": "no",
                },
                {
                    "question": f"Is there more than one {plural_object_name} in the image?",
                    "choices": ["yes", "no"],
                    "answer": "no",
                },
            ]
        )

    return qa_pairs


def extract_spatial_direction_from_caption(caption: str):
    words = caption.split()
    positions = {direction: words.index(direction) for direction in VALID_SPATIAL_DIRECTIONS if direction in words}
    if not positions:
        return None
    return min(positions, key=positions.get)
