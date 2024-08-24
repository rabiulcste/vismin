import argparse
import json
import os
import random
import re
import uuid

import numpy as np
import PIL
import spacy
import torch
from diffusers import (AutoPipelineForText2Image,
                       StableDiffusionGLIGENTextImagePipeline)
from groundingdino.models import build_model
from groundingdino.util.inference import (annotate, load_image, load_model,
                                          predict)
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm

from commons.constants import VALID_COUNTS, VIM_DATA_DIR
from commons.logger import Logger
from commons.utils import load_json_data, set_random_seed

from .filters.vqa_models import VQAModelForTifa
from .utils.helpers import pluralize
from .utils.rmvany_utils import (ObjectCountParser, filter_boxes_optimized,
                                 get_new_caption_for_updated_count,
                                 get_obj_removal_verify_qa_pairs_by_name)
from .utils.spatial_relation_utils import (
    get_obj_existence_verify_qa_pairs_by_name, get_xyxy_from_xywh,
    load_llama_model_for_inpainting, normalize_bounding_box,
    remove_object_from_image_using_llama_model)

logger = Logger.get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed()


class CountingObjectRemovedImgGenerator:
    # Mapping of number words to their numeric values
    NUM_TO_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def __init__(self, vqa_scorer: VQAModelForTifa):
        self.vqa_scorer = vqa_scorer
        self.obj_cnt_parser = ObjectCountParser()
        self.nlp_model = spacy.load("en_core_web_sm")

        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swint_ogc.pth"
        ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
        self.dino_model = self.load_model_hf(
            repo_id=ckpt_repo_id, filename=ckpt_filename, ckpt_config_filename=ckpt_config_filename
        )
        self.gligen_inpaint_pipe = self.load_gligen_inpaint_pipe()
        self.sdxl_pipe = self.load_sdxl_t2i_pipe()

    def load_gligen_inpaint_pipe(self):
        # Insert objects described by image at the region defined by bounding boxes
        pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
            "anhnct/Gligen_Inpainting_Text_Image", torch_dtype=torch.float16
        )
        pipe = pipe.to(device)
        return pipe

    def load_sdxl_t2i_pipe(self):
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.enable_model_cpu_offload()
        return pipe

    def load_model_hf(self, repo_id, filename, ckpt_config_filename):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location="cpu")
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        logger.info(f"Model loaded from {cache_file} \n => {log}")
        return model

    def generate_grounding(self, image, text_prompt, box_threshold=0.2, text_threshold=0.2):
        _, transformed_image = load_image(image)
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=transformed_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cpu",
        )
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
        logger.debug(f"no of boxes: {len(boxes)}, phrases: {phrases}, boxes: {boxes}")
        return boxes_xyxy

    def prepare_edit_annotation_data(self, annotations):
        bounding_box_data = annotations["bounding_boxes"]
        bounding_boxes = [bbox[1] for bbox in bounding_box_data]
        edit_instructions = annotations.get("edit_instructions", [])
        old_edited_phrases = [info["edited_phrase"] for info in edit_instructions]
        most_isolated_indices = self.find_most_isolated_bboxes(bounding_boxes)

        for i in range(len(most_isolated_indices)):
            curr_indices = most_isolated_indices[: i + 1]
            object_names = [bounding_box_data[idx][0] for idx in curr_indices]
            most_isolated_bboxes = [bounding_box_data[idx][1] for idx in curr_indices]
            if len(set(object_names)) == 1:
                object_count_rmv = len(curr_indices)
                object_name = object_names[0]
                input_phrase, edited_phrase, edited_caption = self.get_edited_caption(
                    annotations, object_name, object_count_rmv
                )
                if edited_phrase in old_edited_phrases:
                    print(f"skipping as edited phrase already exists: {edited_phrase}")
                    continue
                if edited_caption is None:
                    continue

                curr_info = {
                    "remove_object_names": object_names,
                    "remove_object_bbox_indices": most_isolated_bboxes,
                    "input_phrase": input_phrase,
                    "edited_phrase": edited_phrase,
                    "edited_caption": edited_caption,
                    "edit_id": str(uuid.uuid4())[:6],
                }
                edit_instructions.append(curr_info)

        return edit_instructions

    @staticmethod
    def unnormalize_bboxes(bboxes, img_width, img_height):
        return [
            [bbox[0] * img_width, bbox[1] * img_height, bbox[2] * img_width, bbox[3] * img_height] for bbox in bboxes
        ]

    def insert_objects_by_image_crop_at_roi_bbox(self, data, num_repeats_gen_from_scratch=2):
        input_image = data["image"]
        prompt = data["prompt"]
        bboxes = [bbox[1] for bbox in data["bounding_box_data"]]
        phrases = [bbox[0] for bbox in data["bounding_box_data"]]
        w, h = input_image.size
        normalized_boxes = [normalize_bounding_box(box, h, w) for box in bboxes]
        gligen_images = data["object_images"]

        generated_images = []
        for _ in range(num_repeats_gen_from_scratch):
            num_inference_steps = random.choice(range(40, 70, 5))
            schedule_sampling_beta = random.choice([x / 10 for x in range(3, 6)])
            guidance_scale = random.choice([7.5, 8.0, 8.5, 9.0, 9.5, 10.0])

            edited_images = self.gligen_inpaint_pipe(
                prompt=prompt,
                gligen_phrases=phrases,
                gligen_inpaint_image=input_image,
                gligen_boxes=normalized_boxes,
                gligen_images=gligen_images,
                gligen_scheduled_sampling_beta=schedule_sampling_beta,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=3,
            ).images
            generated_images.extend(edited_images)
        random.shuffle(generated_images)
        return generated_images

    def verify_original_image(self, image, bounding_box_data):
        tifa_result_dict_final = {"tifa_score": [], "question_details": []}
        for i in range(len(bounding_box_data)):
            object_name, bounding_box = bounding_box_data[i]
            # get all the bounding boxes except the current one
            object_roi_image = image.crop(get_xyxy_from_xywh(bounding_box))
            if object_roi_image.size[0] < 224 or object_roi_image.size[1] < 224:
                object_roi_image = object_roi_image.resize((224, 224), PIL.Image.ANTIALIAS)
            choices = [object_name, "something else"]
            question_answer_pairs = get_obj_existence_verify_qa_pairs_by_name(object_name, choices)
            tifa_result_dict = vqa_scorer.get_tifa_score(question_answer_pairs, object_roi_image, enable_logging=False)
            tifa_result_dict_final["tifa_score"].append(tifa_result_dict["tifa_score"])
            tifa_result_dict_final["question_details"].append(tifa_result_dict["question_details"])

        tifa_result_dict_final["tifa_score"] = sum(tifa_result_dict_final["tifa_score"]) / len(
            tifa_result_dict_final["tifa_score"]
        )
        return tifa_result_dict_final

    def repaint_original_bad_image(self, inp_image, annotations, llama_model, inpaint_img_with_builded_lama):
        inp_image_path = annotations["generated_images"][0]
        generated_image = None

        # we'll re-evaluate the image
        tifa_result_dict_final = self.verify_original_image(inp_image, annotations["bounding_boxes"])
        if tifa_result_dict_final["tifa_score"] == 1:
            generated_image = inp_image

        # repainting the image
        else:
            bounding_box_data = annotations["bounding_boxes"]
            object_names = [bbox[0] for bbox in bounding_box_data]
            bounding_boxes = [bbox[1] for bbox in bounding_box_data]
            # increase bounding box size by 10%, xywh format
            bounding_boxes_ = [[bbox[0], bbox[1], bbox[2] * (1 + 0.1), bbox[3] * (1 + 0.1)] for bbox in bounding_boxes]

            object_names_unique = list(set(object_names))
            objec_images_unique = self.sdxl_pipe(
                prompt=object_names_unique,
                num_inference_steps=1,
                guidance_scale=0.0,
                num_images_per_prompt=1,
                output_type="pil",
            ).images

            # we need the same number of object images as the number of object names
            object_images = []
            for obj_name in object_names:
                obj_idx = object_names_unique.index(obj_name)
                object_images.append(objec_images_unique[obj_idx])

            cleaned_image = remove_object_from_image_using_llama_model(
                llama_model,
                inp_image,
                bounding_boxes_,
                inpaint_img_with_builded_lama,
                dilate_kernel_size=30,
            )
            data = {
                "image": cleaned_image,
                "prompt": annotations["prompt"],
                "bounding_box_data": bounding_box_data,
                "object_images": object_images,
            }
            generated_images = self.insert_objects_by_image_crop_at_roi_bbox(data, num_repeats_gen_from_scratch=1)
            for image in generated_images:
                tifa_result_dict_final = self.verify_original_image(image, bounding_box_data)
                generated_image = image
                if tifa_result_dict_final["tifa_score"] == 1:
                    break

            # save the image on the input image path
            generated_image.save(inp_image_path)
            logger.info(f"Saved repainted image at: {inp_image_path}")

        logger.info(f"tifa results: {tifa_result_dict_final['tifa_score']}")
        annotations["scores"][inp_image_path] = tifa_result_dict_final
        annotations["scores"][inp_image_path]["repaint_try_count"] = (
            annotations["scores"][inp_image_path].get("repaint_try_count", 0) + 1
        )

        return generated_image

    def remove_unwanted_objects_from_image(
        self, inp_image, bounding_box_data, llama_model=None, inpaint_img_with_builded_lama=None, kernel_size=None
    ):
        object_names = [bbox[0] for bbox in bounding_box_data]
        bounding_boxes = [bbox[1] for bbox in bounding_box_data]
        object_names = list(set(object_names))

        # let's do object detection on the cleaned image
        all_bboxes = []
        for object_name in object_names:
            k_bboxes = self.generate_grounding(inp_image, object_name)
            k_bboxes = k_bboxes.tolist()
            all_bboxes.extend(k_bboxes)

        all_bboxes = self.unnormalize_bboxes(all_bboxes, inp_image.width, inp_image.height)
        filtered_bboxes = filter_boxes_optimized(all_bboxes, bounding_boxes)

        if len(filtered_bboxes) and len(all_bboxes) > len(bounding_boxes):
            logger.warning(f"Attempting cleanup of unwanted objects")
            inp_image = remove_object_from_image_using_llama_model(
                llama_model,
                inp_image,
                filtered_bboxes,
                inpaint_img_with_builded_lama,
                dilate_kernel_size=kernel_size,
            )

        return inp_image

    def find_the_count_from_caption(self, caption):
        caption = caption.lower()
        count = 0
        # Split the caption into words and process each word
        words = caption.split()
        for word in words:
            # Check if the word is a number word and add its value
            if word in self.NUM_TO_WORDS:
                count += self.NUM_TO_WORDS[word]
            # Check if the word is a numeral and add its value
            elif word.isdigit():
                count += int(word)
        return count

    def run(self, image_data_dir):
        llama_model, inpaint_img_with_builded_lama = load_llama_model_for_inpainting()

        image_directories = os.listdir(image_data_dir)
        random.shuffle(image_directories)
        logger.info(f"Total number of images found: {len(image_directories)}")

        debug_dir = "diffused_generator/output/counting_debug"
        if os.path.exists(debug_dir):
            import shutil

            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)

        rmv_tot_processed, rpt_tot_processed = 0, 0
        rmv_success_count, repaint_success_cnt = 0, 0
        for image_id in tqdm(image_directories, desc="Removing objects"):
            annotation_fpath = os.path.join(image_data_dir, image_id, "annotations.json")
            data = load_json_data(annotation_fpath)
            if data is None:
                continue

            annotations = data["annotations"]
            prompt = annotations["prompt"]
            inp_image_path = annotations["generated_images"][0]
            inp_image = Image.open(inp_image_path)

            count_in_caption = self.find_the_count_from_caption(prompt)
            count_in_caption_txt = self.obj_cnt_parser.DIGIT_TO_TEXT.get(str(count_in_caption), count_in_caption)
            if all(count != count_in_caption_txt for count in VALID_COUNTS):
                log_msg = (
                    f"Count in caption ({count_in_caption_txt}) does not match "
                    f"any valid counts {VALID_COUNTS[0]} to {VALID_COUNTS[-1]}"
                )
                logger.info(log_msg)
                continue

            # check whether bounding box is equal or less than the count found in the caption
            if len(annotations["bounding_boxes"]) > 2 and count_in_caption < len(annotations["bounding_boxes"]):
                log_msg = (
                    f"Skipping due to count mismatch. "
                    f"Prompt: {prompt}. "
                    f"Bounding box total: {len(annotations['bounding_boxes'])}"
                )
                logger.info(log_msg)
                continue

            if annotations["scores"][inp_image_path]["tifa_score"] != 1:
                if annotations["scores"][inp_image_path].get("repaint_try_count", 0):  # max 1 try
                    logger.info(
                        f"Skipping image as it failed to pass vqa-based object detection filtering after 1 repaint tries"
                    )
                    continue
                logger.info(f"Repainting image: {image_id} as it did not pass vqa-based object detection filtering")

                # remove unwanted objects from the image that pollute the true count
                # inp_image = self.remove_unwanted_objects_from_image(inp_image, annotations["bounding_boxes"], llama_model, inpaint_img_with_builded_lama, 30)
                inp_image = self.repaint_original_bad_image(
                    inp_image, annotations, llama_model, inpaint_img_with_builded_lama
                )
                rpt_tot_processed += 1
                repaint_success_cnt += annotations["scores"][inp_image_path]["tifa_score"] == 1

                with open(annotation_fpath, "w") as f:
                    json.dump(data, f, indent=4)

                logger.info(f"Saved annotations at: {annotation_fpath}")
                logger.info(f"Perc. success (repaint): {repaint_success_cnt / rpt_tot_processed * 100:.2f}%")

            if annotations["scores"][inp_image_path]["tifa_score"] != 1:
                logger.info("Skipping image as input image did not pass vqa-based object detection filtering")
                continue

            if len(annotations["bounding_boxes"]) < 3:
                continue

            limit = len(annotations["bounding_boxes"]) // 2

            edit_instructions = annotations.get("edit_instructions")
            if edit_instructions is None or len(edit_instructions) < limit:
                edit_instructions = self.prepare_edit_annotation_data(annotations)
                annotations["edit_instructions"] = edit_instructions

            annotation_changed = False
            print(f"DEBUG: Processing image {image_id}")
            for info in edit_instructions:
                edit_id = info["edit_id"]
                object_name = info["remove_object_names"][0]
                most_isolated_bboxes = info["remove_object_bbox_indices"]
                edited_image_path = os.path.join(image_data_dir, image_id, f"{edit_id}.png")

                if os.path.exists(edited_image_path) and annotations["scores"].get(edited_image_path, {}).get(
                    "tifa_score"
                ):
                    logger.info(f"Skipping image as negative image already exists at: {edited_image_path}")
                    continue

                best_image = None
                best_tifa_result_dict = {"tifa_score": -1}
                for kernel_size in [30, 25, 20, 15, 10]:
                    # increase bounding box size by 10%, xywh format
                    most_isolated_bboxes = [
                        [bbox[0] - 0.05 * bbox[2], bbox[1] - 0.05 * bbox[3], bbox[2] * 1.1, bbox[3] * 1.1]
                        for bbox in most_isolated_bboxes
                    ]
                    edited_image = remove_object_from_image_using_llama_model(
                        llama_model,
                        inp_image,
                        most_isolated_bboxes,
                        inpaint_img_with_builded_lama,
                        dilate_kernel_size=kernel_size,
                    )

                    tifa_result_dict_final = {"tifa_score": [], "question_details": []}
                    for bbox in most_isolated_bboxes:
                        edited_image_crop = edited_image.crop(get_xyxy_from_xywh(bbox))
                        if edited_image_crop.size[0] < 224 or edited_image_crop.size[1] < 224:
                            edited_image_crop = edited_image_crop.resize((224, 224), PIL.Image.ANTIALIAS)

                        # edited_image_crop.save(f"diffused_generator/output/counting_debug/{object_name}.png")
                        question_answer_pairs = get_obj_removal_verify_qa_pairs_by_name(object_name)
                        tifa_result_dict = self.vqa_scorer.get_tifa_score(
                            question_answer_pairs, edited_image_crop, enable_logging=False
                        )
                        tifa_result_dict_final["tifa_score"].append(tifa_result_dict["tifa_score"])
                        tifa_result_dict_final["question_details"].append(tifa_result_dict["question_details"])
                        print(f"DEBUG: {object_name} => {tifa_result_dict['tifa_score']}")

                    tifa_result_dict_final["tifa_score"] = sum(tifa_result_dict_final["tifa_score"]) / len(
                        tifa_result_dict_final["tifa_score"]
                    )
                    if tifa_result_dict_final["tifa_score"] > best_tifa_result_dict["tifa_score"]:
                        best_tifa_result_dict = tifa_result_dict_final
                        best_image = edited_image

                    if tifa_result_dict_final["tifa_score"] == 1:
                        rmv_success_count += 1
                        break
                if best_image is None:
                    logger.info(f"Skipping image as negative image could not be generated")
                    continue

                best_image.save(edited_image_path)
                logger.info(f"tifa results: {best_tifa_result_dict['tifa_score']}")
                logger.info(f"Saved edited (negative) image at: {edited_image_path}")

                annotations["scores"][edited_image_path] = best_tifa_result_dict
                rmv_tot_processed += 1
                annotation_changed = True

            if annotation_changed:
                with open(annotation_fpath, "w") as f:
                    json.dump(data, f, indent=4)
                logger.info(f"Saved annotations at: {annotation_fpath}")
                logger.info(f"Perc. success (rmv): {rmv_success_count / rmv_tot_processed * 100:.2f}%")

    def get_edited_caption(self, annotations, object_name, reduce_count: int):
        article_match = re.match(r"^(a|an) (.+)$", object_name)
        if article_match:
            print(f"{object_name} -> {article_match.groups()}")
            object_name = article_match.groups()[1]

        caption_text = annotations["prompt"]
        noun_chunks, _ = self.obj_cnt_parser._parse_objects_from_caption(caption_text)
        objects_to_detect = [self.obj_cnt_parser._extract_count_from_chunk(chunk) for chunk in noun_chunks]
        # print(f"DEBUG # {caption_text} / {object_name} => objects_to_detect: {objects_to_detect}")

        original_count_phrase, updated_count_phrase, edited_prompt = None, None, None
        for obj_name, obj_cnt_n, obj_cnt_t in objects_to_detect:
            if obj_name in {object_name, pluralize(object_name)} and obj_cnt_n is not None:
                obj_cnt_n, obj_cnt_t = obj_cnt_n, obj_cnt_t
                obj_cnt_n_upd = obj_cnt_n - reduce_count
                obj_cnt_t_up = self.obj_cnt_parser.DIGIT_TO_TEXT.get(str(obj_cnt_n_upd))
                if str(obj_cnt_n) in caption_text:
                    original_count_phrase = (
                        f"{obj_cnt_n} {pluralize(object_name)}" if obj_cnt_n > 1 else f"{obj_cnt_n} {object_name}"
                    )
                    updated_count_phrase = (
                        f"{obj_cnt_n_upd} {pluralize(object_name)}"
                        if obj_cnt_n_upd > 1
                        else f"{obj_cnt_n_upd} {object_name}"
                    )
                elif obj_cnt_t in caption_text:
                    original_count_phrase = (
                        f"{obj_cnt_t} {pluralize(object_name)}" if obj_cnt_n > 1 else f"{obj_cnt_t} {object_name}"
                    )
                    updated_count_phrase = (
                        f"{obj_cnt_t_up} {pluralize(object_name)}"
                        if obj_cnt_n_upd > 1
                        else f"{obj_cnt_t_up} {object_name}"
                    )

                if updated_count_phrase is not None:
                    edited_prompt = get_new_caption_for_updated_count(
                        annotations["prompt"], original_count_phrase, updated_count_phrase
                    )
                    # print(f"DEBUG # {original_count_phrase} => {updated_count_phrase}")
                    # print(f"DEBUG # {caption_text} => {edited_prompt}")
                    return original_count_phrase.lower(), updated_count_phrase.lower(), edited_prompt

        return None, None, None

    def find_most_isolated_bboxes(self, bounding_boxes, isolation_threshold=50):
        min_distance_to_nearest_neighbor = [float("inf")] * len(bounding_boxes)
        most_isolated_indices = []

        # Calculate the distance between the centers of each pair of bounding boxes
        for i, bbox_i in enumerate(bounding_boxes):
            for j, bbox_j in enumerate(bounding_boxes):
                if i != j:
                    distance = self.center_distance(bbox_i, bbox_j)
                    # Update the minimum distance to the nearest neighbor for bbox_i
                    min_distance_to_nearest_neighbor[i] = min(min_distance_to_nearest_neighbor[i], distance)
        # print(f"min distance to nearest neighbor: {min_distance_to_nearest_neighbor}")
        # Identify the indices of the bounding boxes with the minimum distance to any other above the threshold
        for i, distance in enumerate(min_distance_to_nearest_neighbor):
            if distance >= isolation_threshold:
                most_isolated_indices.append(i)

        # maximum isolated indices can be 3 or at least less than 1 from the  size of bounding boxes
        limit = min(3, len(bounding_boxes) - 1)
        return most_isolated_indices[:limit]

    @staticmethod
    def center_distance(bbox1, bbox2):
        center_x1, center_y1 = bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2
        center_x2, center_y2 = bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2
        return ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use. Possible values: train, validation, test.",
    )
    args = parser.parse_args()
    split = args.split

    args_dict_formatted = ", ".join([f"`{attr}`: {getattr(args, attr)}" for attr in vars(args)])
    logger.info(f"Arguments: ({args_dict_formatted})")

    layout_image_data_dir = os.path.join(VIM_DATA_DIR, f"bootstrap_layout_counting_{split}")
    model_name_or_path = "HuggingFaceM4/idefics2-8b"  # or liuhaotian/llava-v1.6-34b or vikhyatk/moondream2 or llava-hf/llava-v1.6-mistral-7b-hf
    vqa_scorer = VQAModelForTifa(model_name_or_path, load_in_Nbit=None)

    count_rmv_gen = CountingObjectRemovedImgGenerator(vqa_scorer)
    count_rmv_gen.run(layout_image_data_dir)
