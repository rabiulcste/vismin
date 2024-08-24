# this script is so ugly, but it works for now

import argparse
import bdb
import json
import os
import random
import re
import string
import sys
import time
import traceback
import uuid
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from commons.constants import (LANGUAGE_MODEL_NAMES, SYNTH_DIFFUSE_DATA_DIR,
                               SYNTH_ONLY_CATEGORIES, VALID_COUNTS,
                               VIM_DATA_DIR)
from commons.logger import Logger
from commons.utils import CustomJsonEncoder, FileLocker, set_random_seed

from .filters.vqa_models import VQAModelForTifa
from .utils.helpers import get_xyxy_from_xywh
from .utils.llm_utils import GroundedLLM
from .utils.rmvany_utils import SDXLInpainting
from .utils.spatial_relation_utils import (
    adjust_boxes_touching_edges, blank_out_image_regions_by_bounding_boxes,
    create_mask_image_from_bbox, extract_spatial_direction_from_caption,
    get_obj_existence_verify_qa_pairs_by_name, load_llama_model_for_inpainting,
    remove_object_from_image_using_llama_model)

lmd_path = str(Path(__file__).resolve().parent.parent / "llm_grounded_diffusion")
if Path(lmd_path).is_dir() and lmd_path not in sys.path:
    print(f"Adding {lmd_path} to sys.path")
    sys.path.insert(0, lmd_path)


import llm_grounded_diffusion.models as llm_models
import llm_grounded_diffusion.utils as llm_utils
import llm_grounded_diffusion.utils.latents as llm_latents

sys.modules["utils"] = llm_utils
sys.modules["utils.latents"] = llm_latents
sys.modules["models"] = llm_models

import llm_grounded_diffusion.generation.sdxl_refinement as sdxl
from llm_grounded_diffusion import models
from llm_grounded_diffusion.models import sam
from llm_grounded_diffusion.utils import parse
from llm_grounded_diffusion.utils.parse import filter_boxes, show_boxes

logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed()


float_args = [
    "frozen_step_ratio",
    "loss_threshold",
    "ref_ca_loss_weight",
    "fg_top_p",
    "bg_top_p",
    "overall_fg_top_p",
    "overall_bg_top_p",
    "fg_weight",
    "bg_weight",
    "overall_fg_weight",
    "overall_bg_weight",
    "overall_loss_threshold",
    "fg_blending_ratio",
    "mask_th_for_point",
    "so_floor_padding",
]


int_args = [
    "loss_scale",
    "max_iter",
    "max_index_step",
    "overall_max_iter",
    "overall_max_index_step",
    "overall_loss_scale",
    # Set to 0 to disable and set to 1 to enable
    "horizontal_shift_only",
    "so_horizontal_center_only",
    # Set to 0 to disable and set to 1 to enable (default: see the default value in each generation file):
    "use_autocast",
    # Set to 0 to disable and set to 1 to enable
    "use_ref_ca",
]

str_args = ["so_vertical_placement"]


class LlmGroDiffusionProcessor:
    def __init__(
        self,
        llm: GroundedLLM,
        vqa_scorer: VQAModelForTifa,
        prompt_type: str,
        cache_fpath: str,
        device: str = "cuda",
    ):
        self.llm = llm
        self.vqa_scorer = vqa_scorer
        self.prompt_type = prompt_type
        self.device = device
        self.cache_fpath = cache_fpath
        self.generated_edit_instructions = self.load_cached_edit_instructions()
        self.file_locker = FileLocker()

    def load_cached_edit_instructions(self):
        logger.info(f"Loading LLM responses from cache {self.cache_fpath}")
        try:
            with open(self.cache_fpath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find cached edit instructions at {self.cache_fpath}")

    def run(self, args):
        is_notebook = False
        repeats = args.repeats
        scale_boxes_default = not args.no_scale_boxes_default

        run_kwargs = {}
        argnames = float_args + int_args + str_args
        for argname in argnames:
            argvalue = getattr(args, argname)
            if argvalue is not None:
                run_kwargs[argname] = argvalue
                logger.info(f"**Setting {argname} to {argvalue}**")

        if args.no_center_or_align:
            run_kwargs["align_with_overall_bboxes"] = False
            run_kwargs["so_center_box"] = False

        our_models = ["lmd", "lmd_plus"]
        gligen_models = ["gligen", "lmd_plus"]

        # MultiDiffusion will load its own model instead of using the model loaded with `load_sd`.
        custom_models = ["multidiffusion"]

        if args.use_sdv2:
            assert args.run_model not in gligen_models, "gligen only supports SDv1.4"
            # We abbreviate v2.1 as v2
            models.sd_key = "stabilityai/stable-diffusion-2-1-base"
            models.sd_version = "sdv2"
        else:
            if args.run_model in gligen_models:
                models.sd_key = "gligen/diffusers-generation-text-box"
                models.sd_version = "sdv1.4"
            else:
                models.sd_key = "runwayml/stable-diffusion-v1-5"
                models.sd_version = "sdv1.5"

        logger.info(f"Using SD: {models.sd_key}")
        if args.run_model not in custom_models:
            models.model_dict = models.load_sd(
                key=models.sd_key,
                use_fp16=False,
                scheduler_cls=diffusers.schedulers.__dict__[args.scheduler] if args.scheduler else None,
            )
            logger.debug(f"models.model_dict.keys() = {models.model_dict.keys()}")

        if args.run_model in our_models:
            sam_model_dict = sam.load_sam()
            models.model_dict.update(sam_model_dict)

        if not args.dry_run:
            if args.run_model == "lmd":
                import llm_grounded_diffusion.generation.lmd as generation
            elif args.run_model == "lmd_plus":
                import llm_grounded_diffusion.generation.lmd_plus as generation
            elif args.run_model == "sd":
                if not args.ignore_negative_prompt:
                    logger.info(
                        "**You are running SD without `ignore_negative_prompt`. This means that it still uses part of the LLM output and is not a real SD baseline that takes only the prompt."
                    )
                import llm_grounded_diffusion.generation.stable_diffusion_generate as generation
            elif args.run_model == "multidiffusion":
                import llm_grounded_diffusion.generation.multidiffusion as generation
            elif args.run_model == "backward_guidance":
                import llm_grounded_diffusion.generation.backward_guidance as generation
            elif args.run_model == "boxdiff":
                import llm_grounded_diffusion.generation.boxdiff as generation
            elif args.run_model == "gligen":
                import llm_grounded_diffusion.generation.gligen as generation
            else:
                raise ValueError(f"Unknown model type: {args.run_model}")

            # Sanity check: the version in the imported module should match the `run_model`
            version = generation.version
            assert version == args.run_model, f"{version} != {args.run_model}"
            run = generation.run
            if args.use_sdv2:
                version = f"{version}_sdv2"

            # ugly hack to pass the model_dict to the generation script
            generation.sd_key = models.sd_key
            generation.model_dict = models.model_dict
            (
                generation.vae,
                generation.tokenizer,
                generation.text_encoder,
                generation.unet,
                generation.scheduler,
                generation.dtype,
            ) = (
                models.model_dict.vae,
                models.model_dict.tokenizer,
                models.model_dict.text_encoder,
                models.model_dict.unet,
                models.model_dict.scheduler,
                models.model_dict.dtype,
            )

        else:
            version = "dry_run"
            run = None
            generation = argparse.Namespace()

        if args.sdxl:
            # Offload model saves GPU memory.
            sdxl.init(offload_model=True)

        llama_model, inpaint_img_with_builded_lama = load_llama_model_for_inpainting()
        if args.sdxl_retouch_by_mask:
            sdxl_retouch = SDXLInpainting()
            sdxl_retouch.load_model()

        image_ids = list(self.generated_edit_instructions.keys())
        logger.info(f"Total number of annotations: {len(image_ids)}")
        image_ids_after_caching = [
            image_id
            for image_id in image_ids
            if not os.path.exists(
                os.path.join(
                    self.get_output_dir_path_by_image_id(image_id, args.dataset, args.split), "annotations.json"
                )
            )
        ]
        logger.info(f"Number of images already cached {len(image_ids) - len(image_ids_after_caching)}")

        # load qa annotations
        qa_fpath = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"qa_annotations_{args.dataset}_{args.split}.json"
        )
        if not os.path.exists(qa_fpath):
            raise FileNotFoundError(f"Could not find qa annotations at {qa_fpath}")
        with open(qa_fpath, "r") as f:
            qa_annotations = json.load(f)
        # extended_phrase_fpath = os.path.join(
        #     SYNTH_DIFFUSE_DATA_DIR,
        #     "prompt_resources",
        #     f"llm_edits_{args.dataset}",
        #     "mistralaimixtral8x7binstructv0.1",
        #     f"phrase_enhanced_{args.split}.json",
        # )
        # if not os.path.exists(extended_phrase_fpath):
        #     raise FileNotFoundError(f"Could not find extended phrase annotations at {extended_phrase_fpath}")
        # with open(extended_phrase_fpath, "r") as f:
        #     extended_phrase_annotations = json.load(f)

        image_ids_with_qa_ann = [image_id for image_id in image_ids_after_caching if image_id in qa_annotations]
        random.shuffle(image_ids_with_qa_ann)
        logger.info(f"Missing qa annotations for {len(image_ids_after_caching) - len(image_ids_with_qa_ann)} images")
        logger.info(f"Total number of images remaining to be processed: {len(image_ids_with_qa_ann)}")

        tot_processed, success_count = 0, 0
        for image_id in tqdm(image_ids_with_qa_ann, desc="Editing images"):
            kwargs = {}
            scale_boxes = kwargs.get("scale_boxes", scale_boxes_default)
            scale_boxes = False
            # Load from cache
            llm_response = self.generated_edit_instructions.get(str(image_id), None)
            if llm_response is None:
                logger.debug(f"***No cached response for image_id: {image_id}***")
                continue

            logger.debug(f"resp: {llm_response}***")
            logger.debug(f"scale_boxes: {scale_boxes}***")

            output_dir = self.get_output_dir_path_by_image_id(image_id, args.dataset, args.split)
            annotation_file_path = os.path.join(output_dir, "annotations.json")

            if os.path.exists(annotation_file_path):
                logger.info(f"Skipping {image_id} because it already exists")
                continue

            start_time = time.time()

            parse.img_dir = output_dir
            saved_paths = []
            scores_dict = {}
            generated_image, tifa_result_dict_final = None, None

            try:
                prompt = llm_response["prompt"]
                if isinstance(llm_response["bounding_boxes"], str):
                    input_boxes = eval(llm_response["bounding_boxes"])
                else:
                    input_boxes = llm_response["bounding_boxes"]
                bg_prompt = llm_response["background_prompt"]
                neg_prompt = llm_response.get("negative_prompt", "")
                category = llm_response["category"].split("(")[0]
            except Exception as e:
                logger.info(f"***Error: {e}***")
                continue

            print(f"category: {category}")
            if category not in SYNTH_ONLY_CATEGORIES:
                logger.info(f"***Skipping {image_id} as it is not in the category: {category}***")
                continue
            if category == "relation":
                if extract_spatial_direction_from_caption(prompt) is None:
                    logger.info(
                        f"***Skipping {image_id} as it does not have any direction in the category, found: {prompt}***"
                    )
                    continue
            elif category == "counting":
                if all(count not in prompt.lower() for count in VALID_COUNTS):
                    logger.info(
                        f"***Skipping {image_id} as it does not have any count in the category, found: {prompt}***"
                    )
                    continue

            if neg_prompt is None:
                neg_prompt = ""

            if args.ignore_bg_prompt:
                bg_prompt = ""

            if args.ignore_negative_prompt:
                neg_prompt = ""

            try:
                gen_boxes = filter_boxes(input_boxes, scale_boxes=scale_boxes)
            except Exception as e:
                logger.info(f"***Error: {e}***")
                continue
            gen_boxes = adjust_boxes_touching_edges(gen_boxes, 512, 512, shift_amount=10)

            llm_response["bounding_boxes"] = gen_boxes
            logger.info(llm_response)

            if len(gen_boxes) != len(input_boxes):
                logger.info(f"***Some boxes were filtered out: {len(input_boxes)} -> {len(gen_boxes)}***")
                continue

            if not gen_boxes:  # ignore the prompt if no boxes are found
                logger.info("No boxes found, skipping")
                continue

            # object_names = [bbox[0] for bbox in gen_boxes]
            # # Update gen_boxes with improved phrases
            # improve_phrase_list = [
            #     extended_phrase_annotations.get(image_id, {}).get(obj_name, []) for obj_name in object_names
            # ]
            # if extended_phrase_annotations.get(image_id) is not None:
            #     improve_phrase_list = [[phrase.translate(str.maketrans('', '', string.punctuation)).strip().lower() for phrase in curr_list] for curr_list in improve_phrase_list]
            #     improve_phrase_list_str = [", ".join(curr_list[:1]) for curr_list in improve_phrase_list]
            #     gen_boxes = [
            #         (f"{object_name}({improve_phrase})", bbox)
            #         for (object_name, bbox), improve_phrase in zip(gen_boxes, improve_phrase_list_str)
            #     ]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with self.file_locker.locked(output_dir):
                try:
                    spec = {
                        "prompt": prompt,
                        "gen_boxes": gen_boxes,
                        "bg_prompt": bg_prompt,
                        "extra_neg_prompt": neg_prompt,
                    }

                    for _ in range(repeats):
                        if args.run_model in our_models:
                            # Our models load `extra_neg_prompt` from the spec
                            if args.no_synthetic_prompt:
                                # This is useful when the object relationships cannot be expressed only by bounding boxes.
                                output = run(
                                    spec=spec,
                                    bg_seed=random.randint(0, 1000000),
                                    fg_seed_start=random.randint(0, 1000000),
                                    overall_prompt_override=prompt,
                                    **run_kwargs,
                                )
                            else:
                                # Uses synthetic prompt (handles negation and additional languages better)
                                output = run(
                                    spec=spec,
                                    bg_seed=random.randint(0, 1000000),
                                    fg_seed_start=random.randint(0, 1000000),
                                    **run_kwargs,
                                )
                        elif args.run_model == "sd":
                            output = run(
                                prompt=prompt,
                                seed=random.randint(0, 1000000),
                                extra_neg_prompt=neg_prompt,
                                **run_kwargs,
                            )
                        elif args.run_model == "multidiffusion":
                            output = run(
                                gen_boxes=gen_boxes,
                                bg_prompt=bg_prompt,
                                original_ind_base=random.randint(0, 1000000),
                                bootstrapping=args.multidiffusion_bootstrapping,
                                extra_neg_prompt=neg_prompt,
                                **run_kwargs,
                            )
                        elif args.run_model == "backward_guidance":
                            output = run(
                                spec=spec,
                                bg_seed=random.randint(0, 1000000),
                                **run_kwargs,
                            )
                        elif args.run_model == "boxdiff":
                            output = run(
                                spec=spec,
                                bg_seed=random.randint(0, 1000000),
                                **run_kwargs,
                            )
                        elif args.run_model == "gligen":
                            output = run(
                                spec=spec,
                                bg_seed=random.randint(0, 1000000),
                                **run_kwargs,
                            )

                        generated_image = output.image
                        generated_image = (
                            Image.fromarray(generated_image).convert("RGB")
                            if isinstance(generated_image, np.ndarray)
                            else generated_image
                        )

                        # specific filtering for relation category
                        if category == "relation":
                            bounding_box_data = llm_response["bounding_boxes"]
                            # choices = [bbox[0] for bbox in llm_response["bounding_boxes"]] + ["none"]
                            generated_questions_chunk = qa_annotations[image_id]["generated_questions"]
                            tifa_result_dict_final = {"tifa_score": [], "question_details": []}
                            for i in range(len(bounding_box_data)):
                                object_name, other_object_bbox = bounding_box_data[i][0], bounding_box_data[1 - i][1]
                                # object_crop = image.crop(get_xyxy_from_xywh(object_bbox))
                                object_roi_image = remove_object_from_image_using_llama_model(
                                    llama_model,
                                    generated_image,
                                    [other_object_bbox],
                                    inpaint_img_with_builded_lama,
                                    dilate_kernel_size=30,
                                )
                                # question_answer_pairs = get_obj_existence_verify_qa_pairs_by_name(object_name, choices)
                                question_answer_pairs = generated_questions_chunk[i]
                                tifa_result_dict = vqa_scorer.get_tifa_score(
                                    question_answer_pairs, object_roi_image, enable_logging=False
                                )
                                tifa_result_dict_final["tifa_score"].append(tifa_result_dict["tifa_score"])
                                tifa_result_dict_final["question_details"].append(tifa_result_dict["question_details"])
                            # bouning_boxes = [bbox[1] for bbox in llm_response["bounding_boxes"]]
                            # blank_image = blank_out_image_regions_by_bounding_boxes(generated_image, bouning_boxes)
                            # tifa_result_dict = vqa_scorer.get_tifa_score(
                            #     get_obj_existence_verify_qa_pairs_by_name("none", choices), blank_image
                            # )
                            tifa_result_dict_final["tifa_score"] = sum(tifa_result_dict_final["tifa_score"]) / len(
                                tifa_result_dict_final["tifa_score"]
                            )
                            if tifa_result_dict_final["tifa_score"] == 1:
                                break

                        elif category == "counting":
                            bounding_box_data = llm_response["bounding_boxes"]
                            if args.sdxl_retouch_by_mask:
                                for object_name, bbox in bounding_box_data:
                                    prompt = f"{object_name}, high resolution"
                                    width, height = generated_image.size
                                    mask_image = create_mask_image_from_bbox([bbox], height, width)
                                    mask_image = Image.fromarray(mask_image)
                                    generated_image = generated_image.resize((1024, 1024), Image.LANCZOS)
                                    mask_image = mask_image.resize((1024, 1024), Image.LANCZOS)
                                    generated_image = sdxl_retouch.generate_inpainting(
                                        prompt=prompt,
                                        image=generated_image,
                                        mask_image=mask_image,
                                        num_images_per_prompt=1,
                                        num_inference_steps=20,
                                        guidance_scale=7.5,
                                        strength=0.99,
                                    )[0]
                                    generated_image = generated_image.resize((512, 512), Image.LANCZOS)

                            tifa_result_dict_final = {"tifa_score": [], "question_details": []}
                            for i in range(len(bounding_box_data)):
                                object_name, bounding_box = bounding_box_data[i]
                                # get all the bounding boxes except the current one
                                object_roi_image = generated_image.crop(get_xyxy_from_xywh(bounding_box))
                                # save the image for debugging
                                object_roi_image.save(f"diffused_generator/output/cropped_images/{object_name}.png")
                                choices = [object_name, "something else"]
                                question_answer_pairs = get_obj_existence_verify_qa_pairs_by_name(object_name, choices)
                                tifa_result_dict = vqa_scorer.get_tifa_score(
                                    question_answer_pairs, object_roi_image, enable_logging=False
                                )
                                tifa_result_dict_final["tifa_score"].append(tifa_result_dict["tifa_score"])
                                tifa_result_dict_final["question_details"].append(tifa_result_dict["question_details"])

                            tifa_result_dict_final["tifa_score"] = sum(tifa_result_dict_final["tifa_score"]) / len(
                                tifa_result_dict_final["tifa_score"]
                            )
                            if tifa_result_dict_final["tifa_score"] == 1:
                                break

                except (KeyboardInterrupt, bdb.BdbQuit) as e:
                    logger.info(e)
                    exit()
                except RuntimeError:
                    logger.info("***RuntimeError: might run out of memory, skipping the current one***")
                    logger.info(traceback.format_exc())
                    time.sleep(10)
                except Exception as e:
                    logger.info(f"***Error: {e}***")
                    logger.info(traceback.format_exc())
                    continue

                if generated_image is None:
                    logger.info("***No images generated, skipping***")
                    continue

                if not isinstance(tifa_result_dict_final["tifa_score"], (float, int)):
                    logger.info(f"***tifa_score is not a float or an int: {tifa_result_dict_final['tifa_score']}***")
                    continue

                logger.info(f"tifa results: {tifa_result_dict_final['tifa_score']}")
                if not is_notebook:
                    plt.clf()

                if args.sdxl and tifa_result_dict_final["tifa_score"] == 1:
                    generated_image.save("diffused_generator/output/sdxl_refine/orig.png")
                    generated_image = sdxl.refine(
                        image=generated_image,
                        spec=spec,
                        refine_seed=random.randint(0, 1000000),
                        refinement_step_ratio=args.sdxl_step_ratio,
                    )
                    generated_image = generated_image.resize((512, 512), Image.LANCZOS)
                edited_image_path = self.get_image_save_path(output_dir)
                saved_paths.append(edited_image_path)
                self.save_image(generated_image, edited_image_path)
                scores_dict[edited_image_path] = tifa_result_dict_final

                annotations_dict = {
                    "annotations": {
                        **llm_response,
                        "edit_id": str(uuid.uuid4())[:4],
                        "generated_images": saved_paths,
                        "scores": scores_dict,
                    },
                }
                formatted_saved_paths = "\n".join(f"- {item}" for item in saved_paths[-20:])
                logger.info(f"Saved generated images at:\n{formatted_saved_paths}")
                self.save_generated_image_annotations(annotations_dict, annotation_file_path)

                success_count += tifa_result_dict_final.get("tifa_score", 0) == 1
                tot_processed += 1
                end_time = time.time()
                total_time_taken = end_time - start_time
                logger.info(f"Total time taken to generate {len(saved_paths)} images: {total_time_taken:.2f} seconds")
                logger.info(f"Perc. success: {success_count / tot_processed * 100:.2f}%")

    @staticmethod
    def get_output_dir_path_by_image_id(image_id, dataset: str, split: str):
        base_path = os.path.join(VIM_DATA_DIR, f"bootstrap_layout_{dataset}_{split}", f"{image_id}")
        return base_path

    @staticmethod
    def save_image(image, fname):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(fname)

    @staticmethod
    def save_generated_image_annotations(all_captions_dict: dict, output_file_path: str):
        """
        all_captions_dict: a dictionary of image paths and corresponding captions
        """
        if len(all_captions_dict) == 0:
            return

        if os.path.exists(output_file_path):
            with open(output_file_path, "r") as f:
                existing_captions_dict = json.load(f)
            all_captions_dict.update(existing_captions_dict)

        with open(output_file_path, "w") as f:
            json.dump(all_captions_dict, f, cls=CustomJsonEncoder, indent=4)

        logger.info(f"Saved annotations at {output_file_path}")

    @staticmethod
    def get_image_save_path(output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        idx = 0  # Start from zero each time for each new set
        while True:
            save_path = os.path.join(output_dir, f"{idx}.png")
            annotated_save_path = os.path.splitext(save_path)[0] + "_annotated.png"
            if not os.path.exists(save_path) and not os.path.exists(annotated_save_path):
                break
            idx += 1  # Increment the index and check again
        return save_path

    def clean_up_empty_dirs(self, args):
        for image_id in self.generated_edit_instructions.keys():
            output_dir = self.get_output_dir_path_by_image_id(image_id, args.dataset, args.split)
            if not os.path.exists(output_dir):
                continue
            if not os.listdir(output_dir):
                os.rmdir(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="relation",
        choices=["relation", "counting"],
        help="Dataset to use. Possible values: relation, counting.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use. Possible values: train, validation, test.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="llm_layout",
        help=f"Type of prompt to use. Possible values: instruct_pix2pix. Each choice represents a different strategy for generating prompts.",
    )
    parser.add_argument(
        "--run-model",
        default="lmd_plus",
        choices=[
            "lmd",
            "lmd_plus",
            "sd",
            "multidiffusion",
            "backward_guidance",
            "boxdiff",
            "gligen",
        ],
    )
    parser.add_argument(
        "--language_model_name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        choices=LANGUAGE_MODEL_NAMES,
        help=f"Set pre-trained language model to use. Possible values: {', '.join(LANGUAGE_MODEL_NAMES)}.",
    )
    parser.add_argument(
        "--use-upscale",
        action="store_true",
        default=False,
        help="Flag to determine if images should be upscaled to 4x",
    )
    parser.add_argument(
        "--no-center-or-align",
        action="store_true",
        help="Do not perform per-box generation in the center and then align for overall generation",
    )
    parser.add_argument(
        "--no-scale-boxes-default", action="store_true", help="Do not scale the boxes to fill the scene"
    )
    parser.add_argument(
        "--no-synthetic-prompt",
        action="store_true",
        help="Use the original prompt for overall generation rather than a synthetic prompt ([background prompt] with [objects])",
    )
    parser.add_argument(
        "--ignore-bg-prompt",
        action="store_true",
        help="Ignore the background prompt (set background prompt to an empty str)",
    )
    parser.add_argument(
        "--ignore-negative-prompt", action="store_true", help="Ignore the additional negative prompt generated by LLM"
    )
    parser.add_argument("--repeats", default=1, type=int, help="Number of samples for each prompt")
    parser.add_argument("--use-sdv2", action="store_true")
    parser.add_argument("--scheduler", default=None, type=str)
    parser.add_argument("--dry-run", action="store_true", help="skip the generation")

    for float_arg in float_args:
        parser.add_argument("--" + float_arg, default=None, type=float)
    for int_arg in int_args:
        parser.add_argument("--" + int_arg, default=None, type=int)
    for str_arg in str_args:
        parser.add_argument("--" + str_arg, default=None, type=str)
    parser.add_argument("--multidiffusion_bootstrapping", default=20, type=int)
    parser.add_argument("--sdxl", action="store_true", help="Enable sdxl.")
    parser.add_argument(
        "--sdxl-step-ratio", type=float, default=0.4, help="SDXL step ratio: the higher the stronger the refinement."
    )
    parser.add_argument("--sdxl-retouch-by-mask", action="store_true", help="Use mask to retouch the image using SDXL.")
    parser.add_argument("--llm-phrase-refine", action="store_true", help="Refine the prompt using LLM.")

    args = parser.parse_args()

    dataset_name = args.dataset
    split = args.split
    prompt_type = args.prompt_type
    upscale = args.use_upscale
    language_model_name = args.language_model_name
    language_model_name_str = re.sub(r"[/\-]", "", language_model_name).lower()
    cache_fpath = os.path.join(
        SYNTH_DIFFUSE_DATA_DIR,
        "prompt_resources",
        f"llm_edits_{dataset_name}",
        language_model_name_str,
        f"{prompt_type}_{split}.json",
    )
    model_name_or_path = (
        "llava-hf/llava-v1.6-mistral-7b-hf"  # or liuhaotian/llava-v1.6-34b or liuhaotian/llava-v1.5-13b
    )
    load_in_Nbit = 8
    llm = GroundedLLM(language_model_name=language_model_name, prompt_type=prompt_type, device=device)
    vqa_scorer = VQAModelForTifa(model_name_or_path, load_in_Nbit=load_in_Nbit)
    image_edit_proc = LlmGroDiffusionProcessor(
        llm=llm,
        vqa_scorer=vqa_scorer,
        cache_fpath=cache_fpath,
        prompt_type=prompt_type,
        device=device,
    )
    image_edit_proc.run(args)
    image_edit_proc.clean_up_empty_dirs(args)
