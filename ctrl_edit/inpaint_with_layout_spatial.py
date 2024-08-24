import argparse
import json
import os
import random
import shutil
import time
import uuid
from typing import Union

import cv2
import numpy as np
import torch
from diffusers import (AutoPipelineForText2Image,
                       StableDiffusionGLIGENTextImagePipeline)
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm

from commons.constants import SYNTH_DIFFUSE_DATA_DIR, VIM_DATA_DIR
from commons.logger import Logger
from commons.utils import FileLocker, load_json_data, set_random_seed

from .filters.vqa_models import VQAModelForTifa
from .utils.helpers import get_xyxy_from_xywh
from .utils.spatial_relation_utils import (
    adjust_boxes_touching_edges, extract_spatial_direction_from_caption,
    load_llama_model_for_inpainting, normalize_bounding_box,
    remove_object_from_image_using_llama_model, swap_phrases_in_spatial_prompt)

logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed()


class GligenInpaintByImageBbox:
    def __init__(self, repo_id: str = None):
        self.repo_id = repo_id
        self.gligen_inpaint_pipe = self.load_gligen_inpaint_pipe()

    def load_gligen_inpaint_pipe(self):
        # Insert objects described by image at the region defined by bounding boxes
        pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
            "anhnct/Gligen_Inpainting_Text_Image", torch_dtype=torch.float16
        )
        pipe = pipe.to(device)
        return pipe

    def insert_objects_by_image_crop_at_roi_bbox(self, data, num_repeats_gen_from_scratch=2):
        input_image = data["image"]
        input_image_cleaned = data["image_cleaned"]
        prompt = data["prompt"]
        bboxes = [bbox[1] for bbox in data["bounding_box_data"]]
        swapped_bboxes = [bbox[1] for bbox in data["swapped_bounding_box_data"]]
        phrases = [bbox[0] for bbox in data["bounding_box_data"]]
        w, h = input_image.size
        normalized_boxes = [normalize_bounding_box(box, h, w) for box in swapped_bboxes]
        gligen_images = [input_image.crop(get_xyxy_from_xywh(box)) for box in bboxes]

        generated_images = []
        for _ in range(num_repeats_gen_from_scratch):
            num_inference_steps = random.choice(range(40, 70, 5))
            schedule_sampling_beta = random.choice([x / 10 for x in range(3, 6)])
            guidance_scale = random.choice([7.5, 8.0, 8.5, 9.0, 9.5, 10.0])

            edited_images = self.gligen_inpaint_pipe(
                prompt=prompt,
                gligen_phrases=phrases,
                gligen_inpaint_image=input_image_cleaned,
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
        return generated_images, gligen_images

    def insert_objects_by_image_crop_at_roi_bbox_bad_image(self, data, num_repeats_gen_from_scratch=2):
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


def verify_already_generated_but_unfiltered_images(
    vqa_scorer: VQAModelForTifa,
    image_path: Union[str, Image.Image],
    annotations: dict,
    generated_questions_chunk: list,
    eval_negative_image: bool,
    image=None,
    debug: bool = False,
):
    logger.info(f"Verifying already generated but unfiltered image: {image_path}")
    # also print the prompt
    logger.info(f"Prompt: {annotations['prompt']}")
    bounding_box_data = annotations["bounding_boxes"]

    if image is not None:
        inp_image = image
    else:
        inp_image = load_image(image_path)

    # choices = [bbox[0] for bbox in bounding_box_data] + ["none"]
    tifa_result_dict_final = {"tifa_score": [], "question_details": []}
    for i in range(2):
        if eval_negative_image:
            object_name, object_bbox = bounding_box_data[1 - i][0], bounding_box_data[i][1]
            question_answer_pairs = generated_questions_chunk[1 - i]
        else:
            object_name, object_bbox = bounding_box_data[i]
            question_answer_pairs = generated_questions_chunk[i]
        object_bbox = [
            object_bbox[0] - 0.05 * object_bbox[2],
            object_bbox[1] - 0.05 * object_bbox[3],
            object_bbox[2] * 1.1,
            object_bbox[3] * 1.1,
        ]
        object_crop = inp_image.crop(get_xyxy_from_xywh(object_bbox))
        # question_answer_pairs = get_obj_existence_verify_qa_pairs_by_name(object_name, choices)
        tifa_result_dict = vqa_scorer.get_tifa_score(question_answer_pairs, object_crop)
        tifa_result_dict_final["tifa_score"].append(tifa_result_dict["tifa_score"])
        tifa_result_dict_final["question_details"].append(tifa_result_dict["question_details"])

    if debug:
        uid = str(uuid.uuid4())[:4]
        object_crop.save(f"diffused_generator/output/spatial_relation/{uid}_{object_name}_cropped.png")
        inp_image.save(f"diffused_generator/output/spatial_relation/{uid}_inpainted.png")

    tifa_result_dict_final["tifa_score"] = sum(tifa_result_dict_final["tifa_score"]) / len(
        tifa_result_dict_final["tifa_score"]
    )
    logger.info(f"tifa results: {tifa_result_dict_final['tifa_score']}")
    annotations.setdefault("scores", {}).setdefault(image_path, {}).update(tifa_result_dict_final)
    annotations["scores"][image_path]["repaint_try_cnt"] = (
        annotations["scores"][image_path].get("repaint_try_cnt", 0) + 1
    )


class SRelationEditedImgGenerator:
    def __init__(self, vqa_scorer: VQAModelForTifa, gligen_inpaint_pipe: GligenInpaintByImageBbox):
        self.vqa_scorer = vqa_scorer
        self.gligen_inpaint_pipe = gligen_inpaint_pipe
        self.file_locker = FileLocker()
        self.sdxl_pipe = self.load_sdxl_t2i_pipe()

    def load_sdxl_t2i_pipe(self):
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.enable_model_cpu_offload()
        return pipe

    @staticmethod
    def swap_bounding_boxes_position(bounding_box_data: list):
        original_sizes = [(w, h) for _, (_, _, w, h) in bounding_box_data]
        swapped_bboxes = []
        for i, (name, (x, y, _, _)) in enumerate(bounding_box_data):
            new_x, new_y = bounding_box_data[(i + 1) % len(bounding_box_data)][1][:2]
            w, h = original_sizes[i]
            swapped_bboxes.append((name, (new_x, new_y, w, h)))
        return swapped_bboxes

    def repaint_original_bad_image(
        self, inp_image_path, annotations, generated_questions_chunk, llama_model, inpaint_img_with_builded_lama
    ):
        inp_image = Image.open(inp_image_path).convert("RGB")
        bounding_box_data = annotations["bounding_boxes"]
        object_names = [bbox[0] for bbox in bounding_box_data]
        bounding_boxes = [bbox[1] for bbox in bounding_box_data]
        # increase bounding box size by 10%, xywh format
        bounding_boxes_ = [[bbox[0], bbox[1], bbox[2] * (1 + 0.1), bbox[3] * (1 + 0.1)] for bbox in bounding_boxes]

        object_images = self.sdxl_pipe(
            prompt=object_names, num_inference_steps=1, guidance_scale=0.0, num_images_per_prompt=1, output_type="pil"
        ).images

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
        generated_images = gligen_inpaint_pipe.insert_objects_by_image_crop_at_roi_bbox_bad_image(
            data, num_repeats_gen_from_scratch=1
        )
        # randomly choose one of the generated images
        generated_image = random.choice(generated_images)

        verify_already_generated_but_unfiltered_images(
            vqa_scorer,
            inp_image_path,
            annotations,
            generated_questions_chunk,
            eval_negative_image=False,
            image=generated_image,
        )

        # save the image on the input image path
        generated_image.save(inp_image_path)
        logger.info(f"Saved repainted image at: {inp_image_path}")

    def run(self, image_data_dir: str, split: str, debug: bool = False):
        llama_model, inpaint_img_with_builded_lama = load_llama_model_for_inpainting()

        # load qa annotations
        qa_fpath = os.path.join(SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"qa_annotations_relation_{split}.json")
        if not os.path.exists(qa_fpath):
            raise FileNotFoundError(f"Could not find qa annotations at {qa_fpath}")
        with open(qa_fpath, "r") as f:
            qa_annotations = json.load(f)

        image_directories = os.listdir(image_data_dir)
        random.shuffle(image_directories)
        logger.info(f"Total number of images found: {len(image_directories)}")

        tot_processed, success_count = 0, 0
        tifa_failures = 0
        for image_id in tqdm(image_directories, desc="Inpainting images"):
            curr_dir_path = os.path.join(image_data_dir, image_id)
            annotation_fpath = os.path.join(curr_dir_path, "annotations.json")
            data = load_json_data(annotation_fpath)
            if data is None:
                continue

            annotations = data["annotations"]
            bounding_box_data = annotations["bounding_boxes"]

            if extract_spatial_direction_from_caption(annotations["prompt"]) is None:
                logger.warning("Skipping image as it does not have any valid direction in the prompt")
                continue

            if len(bounding_box_data) != 2:
                logger.warning("Skipping image as it does not have exactly 2 objects")
                continue

            # check the area of first object and second object, if the difference is more than 50% then skip, the values are not bounded by 0 and 1
            area_first = bounding_box_data[0][1][2] * bounding_box_data[0][1][3]
            area_second = bounding_box_data[1][1][2] * bounding_box_data[1][1][3]
            if abs(area_first - area_second) / max(area_first, area_second) > 0.5:
                logger.warning("Skipping image as the area difference is more than 50%")
                continue

            # increase bounding box size by 10%
            bounding_box_data = [
                [phrase, [bbox[0] - 0.05 * bbox[2], bbox[1] - 0.05 * bbox[3], bbox[2] * 1.1, bbox[3] * 1.1]]
                for phrase, bbox in bounding_box_data
            ]

            if image_id not in qa_annotations:
                logger.warning(f"QA annotations not found for image: {image_id}")
                continue
            generated_questions_chunk = qa_annotations[image_id]["generated_questions"]

            inp_image_path = annotations["generated_images"][0]
            if (
                "tifa_score" not in annotations.get("scores", {}).get(inp_image_path, {})
                or annotations["scores"][inp_image_path]["tifa_score"] < 0.95
            ):
                if annotations["scores"][inp_image_path].get("repaint_try_cnt", 0):  # max 1 try
                    logger.warning(f"Skipping image as input image has already been repainted once")
                    continue
                verify_already_generated_but_unfiltered_images(
                    vqa_scorer, inp_image_path, annotations, generated_questions_chunk, eval_negative_image=False
                )
                # repaint the original bad image
                if annotations["scores"][inp_image_path]["tifa_score"] < 0.95:
                    self.repaint_original_bad_image(
                        inp_image_path,
                        annotations,
                        generated_questions_chunk,
                        llama_model,
                        inpaint_img_with_builded_lama,
                    )

                with open(annotation_fpath, "w") as f:
                    json.dump(data, f, indent=4)
                logger.info(f"Saved annotations at: {annotation_fpath}")

            if annotations["scores"][inp_image_path]["tifa_score"] < 0.95:
                logger.warning(
                    f"Skipping image as input image did not pass vqa-based object detection filtering, tifa score = {annotations['scores'][inp_image_path]['tifa_score']}"
                )
                tifa_failures += 1
                continue

            if tifa_failures % 20 == 0:
                logger.info(f"Tifa failure count: {tifa_failures}")

            with self.file_locker.locked(curr_dir_path) as lock_acquired:
                if not lock_acquired:
                    logger.warning(f"Skipping image id: {image_id} as another process is working on it.")
                    continue

                start_time = time.time()
                edited_image_path = inp_image_path.replace(".png", "_negative.png")
                if os.path.exists(edited_image_path):
                    if (
                        "tifa_score" not in annotations.get("scores", {}).get(edited_image_path, {})
                        or 0.85 < annotations["scores"][edited_image_path]["tifa_score"] < 1
                    ):
                        print(f"DEBUG: {inp_image_path} -> {edited_image_path}")
                        if annotations["scores"].get(edited_image_path, {}).get("repaint_try_count", 0):  # max 1 try
                            continue
                        verify_already_generated_but_unfiltered_images(
                            vqa_scorer,
                            edited_image_path,
                            annotations,
                            generated_questions_chunk,
                            eval_negative_image=True,
                        )
                        with open(annotation_fpath, "w") as f:
                            json.dump(data, f, indent=4)
                        logger.info(f"Saved annotations at: {annotation_fpath}")
                    continue

                bounding_boxes = [bbox[1] for bbox in bounding_box_data]
                input_image = Image.open(inp_image_path).convert("RGB")
                # we will remove existing objects from the image to have a clean canvas
                input_image_cleaned = remove_object_from_image_using_llama_model(
                    llama_model, input_image, bounding_boxes, inpaint_img_with_builded_lama, dilate_kernel_size=30
                )
                edited_prompt = swap_phrases_in_spatial_prompt(annotations["prompt"], bounding_box_data)
                if edited_prompt is None:
                    logger.warning("Skipping image as prompt could not be swapped")
                    continue

                # choices = [bbox[0] for bbox in bounding_box_data] + ["none"]
                swapped_bounding_box_data = self.swap_bounding_boxes_position(bounding_box_data)
                swapped_bounding_box_data = adjust_boxes_touching_edges(swapped_bounding_box_data, 512, 512, 10)
                inpainting_data = {
                    "image": input_image,
                    "image_cleaned": input_image_cleaned,
                    "bounding_box_data": bounding_box_data,
                    "swapped_bounding_box_data": swapped_bounding_box_data,
                    "prompt": edited_prompt,
                }
                edited_images, object_crops_used = gligen_inpaint_pipe.insert_objects_by_image_crop_at_roi_bbox(
                    inpainting_data
                )
                # get the max width and height from the bounding boxes, note that bbox are 2d list with [x, y, w, h]
                max_w, max_h = max([bbox[2] for bbox in bounding_boxes]), max([bbox[3] for bbox in bounding_boxes])
                edited_image, tifa_result_dict_final = None, None
                for image in edited_images:
                    tifa_result_dict_final = {"tifa_score": [], "question_details": []}
                    for i in range(2):
                        object_name, object_bbox = bounding_box_data[1 - i][0], bounding_box_data[i][1]
                        object_bbox = [object_bbox[0], object_bbox[1], max_w, max_h]
                        object_crop = image.crop(get_xyxy_from_xywh(object_bbox))

                        # question_answer_pairs = get_obj_existence_verify_qa_pairs_by_name(object_name, choices)
                        question_answer_pairs = generated_questions_chunk[1 - i]
                        tifa_result_dict = vqa_scorer.get_tifa_score(question_answer_pairs, object_crop)
                        tifa_result_dict_final["tifa_score"].append(tifa_result_dict["tifa_score"])
                        tifa_result_dict_final["question_details"].append(tifa_result_dict["question_details"])
                    edited_image = image
                    tifa_result_dict_final["tifa_score"] = sum(tifa_result_dict_final["tifa_score"]) / len(
                        tifa_result_dict_final["tifa_score"]
                    )
                    if tifa_result_dict_final["tifa_score"] == 1:
                        break

                edited_image.save(edited_image_path)
                annotations["edited_prompt"] = edited_prompt
                annotations["scores"][edited_image_path] = tifa_result_dict_final
                logger.info(f"tifa results: {tifa_result_dict_final['tifa_score']}")
                logger.info(f"Saved edited (negative) image at: {edited_image_path}")

                # concatenate the images for debugging purposes
                if debug:
                    combined_img_path = inp_image_path.replace(".png", "_combined.png")
                    edited_image = edited_image.resize(input_image.size, Image.LANCZOS)
                    combined_img = np.hstack((np.array(input_image), np.array(edited_image)))
                    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(combined_img_path, combined_img)

                with open(annotation_fpath, "w") as f:
                    json.dump(data, f, indent=4)
                logger.info(f"Saved annotations at: {annotation_fpath}")
                success_count += tifa_result_dict_final.get("tifa_score", 0) == 1
                tot_processed += 1

                end_time = time.time()
                total_time_taken = end_time - start_time
                logger.info(f"Total time taken to generate hard-negative image: {total_time_taken:.2f} seconds")
                logger.info(f"Perc. success: {success_count / tot_processed * 100:.2f}%")


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

    layout_image_data_dir = os.path.join(VIM_DATA_DIR, f"bootstrap_layout_relation_{split}")
    model_name_or_path = "HuggingFaceM4/idefics2-8b"
    vqa_scorer = VQAModelForTifa(model_name_or_path, load_in_Nbit=4)
    gligen_inpaint_pipe = GligenInpaintByImageBbox()
    gen_proc = SRelationEditedImgGenerator(vqa_scorer, gligen_inpaint_pipe)
    gen_proc.run(layout_image_data_dir, split)
