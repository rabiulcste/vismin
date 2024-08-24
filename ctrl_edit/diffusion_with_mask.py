# some code is borrowed from https://github.com/IDEA-Research/GroundingDINO
import argparse
import json
import os
import random
import re
import time
import warnings
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import torch
from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline
from groundingdino.models import build_model
from groundingdino.util.inference import (annotate, load_image, load_model,
                                          predict)
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

from commons.constants import (LANGUAGE_MODEL_NAMES, SYNTH_DIFFUSE_DATA_DIR,
                               VIM_DATA_DIR)
from commons.logger import Logger
from commons.utils import CustomJsonEncoder, FileLocker, set_random_seed

from .filters.vqa_models import VQAModelForTifa
from .utils.helpers import (annotate_image_with_boxes,
                            get_coco_path_by_image_id,
                            group_outputs_for_batch_repeats,
                            load_coco_captions, load_vsr_captions)

warnings.filterwarnings("ignore")

# Set up logger
logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger.warning(f"Setting tokenizers parallelism to false")


class MaskedInpainter:
    def __init__(
        self,
        repo_id,
        ckpt_filename,
        ckpt_config_filename,
        diffusion_model_name="sd2",
        device="cuda",
    ):
        logger.info(
            f"Initializing {self.__class__.__name__} with repo_id: {repo_id} and ckpt_filename: {ckpt_filename}"
        )
        self.device = device
        self.diffusion_model_name = diffusion_model_name
        self.dino_model = self.load_model_hf(repo_id, ckpt_filename, ckpt_config_filename)
        self.load_diffusion_model()
        self.load_komos2_model_processor()

    def load_diffusion_model(self):
        if self.diffusion_model_name == "sd2":
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
                variant="fp16",
            )
        elif self.diffusion_model_name == "sdxl":
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16",
            )
            # Check if PyTorch version is 2.x
            if int(torch.__version__.split(".")[0]) >= 2:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        self.pipe.to(device)

    def load_model_hf(self, repo_id, filename, ckpt_config_filename):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        args.device = self.device
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location="cpu")
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        logger.info(f"Model loaded from {cache_file} \n => {log}")
        return model

    def load_komos2_model_processor(self):
        model_path = "microsoft/kosmos-2-patch14-224"
        self.kosmos2_model = Kosmos2ForConditionalGeneration.from_pretrained(model_path, device_map="cpu")
        self.kosmos2_processor = AutoProcessor.from_pretrained(model_path)
        self.kosmos2_processor.tokenizer.padding_side = "left"

    def generate_grounding_kosmos2(self, image: PIL.Image.Image, text_prompt):
        _device = self.kosmos2_model.device
        prompt = f"<grounding><phrase> {text_prompt}</phrase>"
        inputs = self.kosmos2_processor(text=prompt, images=image, return_tensors="pt")
        output_ids = self.kosmos2_model.generate(
            pixel_values=inputs["pixel_values"].to(_device),
            input_ids=inputs["input_ids"].to(_device),
            attention_mask=inputs["attention_mask"].to(_device),
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"].to(_device),
            use_cache=True,
            max_new_tokens=64,
        )
        decoded_text = self.kosmos2_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        processed_text, grounded_entities = self.kosmos2_processor.post_process_generation(decoded_text)
        logger.info(f"{text_prompt} => {grounded_entities}")
        return next(
            ((np.array(bbox), phrase) for phrase, indices, bbox in grounded_entities if phrase == text_prompt),
            (None, None),
        )

    def generate_grounding(self, image, text_prompt, box_threshold=0.35, text_threshold=0.25):
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cpu",
        )

        logger.info(f"no of boxes: {len(boxes)}, phrases: {phrases}, boxes: {boxes}")

        # check or remove boxes that capture the entire image
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        boxes_area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

        # Keep only the boxes that cover less than 85% of the image
        boxes_area_tensor = boxes_area
        mask = boxes_area_tensor < 0.85
        indices = np.where(mask)[0].tolist()
        boxes_xyxy = boxes_xyxy[mask]  # making sure boxes is a tensor
        phrases = [phrases[i] for i in indices]

        logger.info(f"no of boxes (post-filtering): {len(boxes_xyxy)}, phrases: {phrases}, boxes: {boxes_xyxy}")
        return boxes_xyxy, phrases

    def generate_masks_with_grounding(self, image: PIL.Image.Image, boxes_xyxy):
        image_np = np.array(image)
        h, w, _ = image_np.shape
        boxes_unnorm = boxes_xyxy * np.array([w, h, w, h])
        logger.debug(f"boxes: {boxes_xyxy} => boxes_unnorm: {boxes_unnorm}")

        mask = np.zeros_like(image_np)
        for box in boxes_unnorm:
            x0, y0, x1, y1 = box
            mask[int(y0) : int(y1), int(x0) : int(x1), :] = 255
        return mask

    def gligen_masked_inpainting(
        self,
        prompt: str,
        image: PIL.Image.Image,
        mask_image: PIL.Image.Image,
        boxes,
        gligen_phrases,
        negative_prompt: str,
        scheduled_sampling_beta: float,
        guidance_scale: float,
        num_inference_steps: int,
        num_images_per_prompt: int,
    ):
        input_image_for_inpaint = image.resize((512, 512))
        xyxy_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").tolist()
        logger.debug(f"debug = prompt: {prompt}, gligen phrases: {gligen_phrases}, xyxy boxes: {xyxy_boxes}")
        image_inpainting = self.pipe(
            prompt=prompt,
            gligen_phrases=gligen_phrases,
            gligen_inpaint_image=input_image_for_inpaint,
            gligen_boxes=xyxy_boxes,
            gligen_scheduled_sampling_beta=scheduled_sampling_beta,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            output_type="pil",
        ).images

        return image_inpainting

    def sd_masked_inpainting(
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


class ImageEditProcessor:
    def __init__(
        self,
        diffuser: MaskedInpainter,
        vqa_scorer: VQAModelForTifa,
        prompt_type: str,
        cache_dir: str,
        dataset: str,
        split: str,
        chunk_index: int,
        device: str,
    ):
        self.file_locker = FileLocker()
        self.diffuser = diffuser
        self.vqa_scorer = vqa_scorer
        self.prompt_type = prompt_type
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.split = split
        self.chunk_index = chunk_index
        self.generated_edit_instructions = self.load_cached_data("edit_instructions")
        self.generated_edit_enhanced_phrases = self.load_cached_data("edit_enhanced_phrases")
        self.generated_llm_filtered_edits = self.load_cached_data("llm_filtered_edits")
        self.generated_qa_annotations = self.load_cached_data("qa_annotations")

    def load_cached_data(self, data_type):
        if data_type == "edit_instructions":
            filename = f"{self.prompt_type}_{self.split}"
        elif data_type == "qa_annotations":
            base_dir = os.path.join(SYNTH_DIFFUSE_DATA_DIR, "prompt_resources")
            filename = f"qa_annotations_{self.dataset}_{self.split}"
        elif data_type == "edit_enhanced_phrases":
            filename = f"phrase_enhanced_{self.split}"
        elif data_type == "llm_filtered_edits":
            filename = f"edit_instruction_filtered_{self.split}"
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        if self.chunk_index is not None:
            filename = f"{filename}_chunked/chunk_{self.chunk_index}.json"
        else:
            filename = f"{filename}.json"

        filepath = os.path.join(self.cache_dir if data_type != "qa_annotations" else base_dir, filename)

        try:
            logger.info(f"Loading cached {data_type} from {filepath}")
            with open(filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            if data_type in ["edit_enhanced_phrases"]:
                return {}  # TODO: remove this line if you want to raise an error instead
            raise FileNotFoundError(f"Could not find cached {data_type} file at {filepath}")

    def get_edit_instruction(self, image_id, image_caption: str, annotation_file_path: str):
        """
        image_id: an input image id
        image_caption: an input image caption
        return:
            edit_instructions: create a list of edit instructions
        """
        # take the intersection of (edit_instruct_llmgen_ids, edit_instruct_llmver_ids)
        edit_instruct_llmgen_data = self.generated_edit_instructions.get(str(image_id), {}).get(image_caption)
        edit_instruct_llmgen_ids = [item["edit_id"] for item in edit_instruct_llmgen_data]
        edit_instruct_llmver_data = self.generated_llm_filtered_edits.get(str(image_id), {})
        edit_instruct_llmver_ids = []
        for edit_id, edit_info in edit_instruct_llmver_data.items():
            if edit_info["reject"] == "NO":
                edit_instruct_llmver_ids.append(edit_id)
        logger.debug(f"LLM suggested outputs: {edit_instruct_llmgen_data}")

        valid_edit_ids_so_far = list(set(edit_instruct_llmgen_ids) & set(edit_instruct_llmver_ids))
        llm_ver_rejected_cnt = len(edit_instruct_llmgen_ids) - len(valid_edit_ids_so_far)
        print(
            f"llm-gen: {edit_instruct_llmgen_ids}, llm-ver: {edit_instruct_llmver_ids}, valid: {valid_edit_ids_so_far}"
        )

        # edit_instruct_llmgen_data after llmver
        edit_instruct_remaning = [
            item for item in edit_instruct_llmgen_data if item["edit_id"] in valid_edit_ids_so_far
        ]
        qa_avail_img_edit_ids = [
            edit_id
            for edit_id in valid_edit_ids_so_far
            if edit_id in self.generated_qa_annotations.get(str(image_id), {})
        ]
        # edit_instruct_llmgen_data after llmver > qaann
        edit_instruct_remaning = [item for item in edit_instruct_remaning if item["edit_id"] in qa_avail_img_edit_ids]
        missing_qa_ann_cnt = len(valid_edit_ids_so_far) - len(edit_instruct_remaning)

        # also make sure edit_ids are not already cached
        already_in_imgen_cache_cnt = 0
        if os.path.exists(annotation_file_path):
            with open(annotation_file_path, "r") as f:
                existing_annotations_dict = json.load(f)
            cached_imgen_edit_ids = [item["edit_id"] for item in existing_annotations_dict.get("annotations", [])]
            # edit_instruct_llmgen_data after llmver > qaann > imgen_cache
            edit_instruct_remaning = [
                item for item in edit_instruct_remaning if item["edit_id"] not in cached_imgen_edit_ids
            ]
            already_in_imgen_cache_cnt = len(cached_imgen_edit_ids)

        # generated_outputs_filtered = [item for item in generated_outputs_filtered if "attribute" in item["category"]]
        logger.info(
            f"LLM edit instructions total: {len(edit_instruct_llmgen_data)}, "
            f"LLM verification rejected total: {llm_ver_rejected_cnt}, "
            f"Missing QA annotation: {missing_qa_ann_cnt}, "
            f"Already found in imgen cache: {already_in_imgen_cache_cnt}, "
            f"Remaining edit instructions: {len(edit_instruct_remaning)}"
        )
        return edit_instruct_remaning

    def image_diffusion_edit_and_rank(
        self, image_id: str, image_path: str, input_caption: str, edits_info: List[Dict[str, str]]
    ):
        """
        image: an input image for diffusion
        edits_info: a list of dictionaries, each containing an edit instruction and a new caption
        return:
            outputs: a list of dictionaries, each containing an edit instruction, a new caption, and a highest-scored generated image
        """
        start_time = time.time()
        num_images_per_prompt = 3  # number of sample images for diffusion
        negative_prompt = (
            "cropped, clipped, invisible, half visible, trimmed, distorted, (unnatural, unreal, unusual), (deformed iris, deformed pupils, "
            "semi-realistic, cgi, 3d, cartoon, anime), (deformed, distorted, disfigured), bad anatomy, "
            "extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, bad quality"
        )

        prompt_list = [f"{info['input_phrase']} => {info['edited_phrase']}" for info in edits_info]
        formatted_prompt_list = "\n".join(f"- {item}" for item in prompt_list[-20:])
        logger.info(f"Sampled prompt list (out of {len(prompt_list)}) :\n{formatted_prompt_list}")

        input_image, transformed_image = load_image(image_path)

        image_data_list = []
        # Generate grounding and prepare image data
        for info in edits_info:
            try:
                grounding_phrase = info["input_phrase"]
                boxes, phrases = self.diffuser.generate_grounding(
                    transformed_image, grounding_phrase, box_threshold=0.25, text_threshold=0.25
                )
                if boxes is None or boxes.shape[0] == 0:
                    boxes, phrases = self.diffuser.generate_grounding_kosmos2(input_image, grounding_phrase)
            except Exception as e:
                logger.error(f"Error in generate_grounding: {e}")

            if boxes is None or boxes.shape[0] == 0:
                continue

            mask_image = self.diffuser.generate_masks_with_grounding(input_image, boxes)
            image_data_list.append(
                {
                    "mask_image": Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image,
                    "boxes": boxes,
                    "edit_info": info,
                }
            )
        input_image = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image

        if not image_data_list:
            logger.warning(f"Could not generate grounding for image id: {image_path}, caption: {input_caption}")
            return []

        # Image generation for each engine
        for data in image_data_list:
            info = data["edit_info"]
            config = {
                "scheduled_sampling_beta": random.choice([x / 10 for x in range(5, 11)]),
                "num_inference_steps": random.choice(range(50, 100, 5)),
                "guidance_scale": random.choice(np.arange(5, 12, 1.5).tolist()),
            }
            use_negative_prompt = random.choice([True, False])

        # Batch processing for engines like 'sdxl'
        batch_size = 1
        for i in range(0, len(image_data_list), batch_size):
            batch_data = image_data_list[i : i + batch_size]
            input_images = [input_image.resize((1024, 1024)) for _ in batch_data]
            mask_images = [data["mask_image"].resize((1024, 1024)) for data in batch_data]

            prompts = []
            for data in batch_data:
                edited_phrase = data["edit_info"]["edited_phrase"]
                edit_id = data["edit_info"]["edit_id"]
                # conditionally add enhanced phrases if they exist to the prompt
                enhanced_phrases = self.generated_edit_enhanced_phrases.get(str(image_id), {}).get(edit_id, [])
                if isinstance(enhanced_phrases, list) and all(isinstance(item, str) for item in enhanced_phrases):
                    enhanced_phrases_str = ", ".join(enhanced_phrases[:3])
                    logger.info(f"{edited_phrase} => enhanced phrases: {enhanced_phrases_str}")
                    edited_phrase = edited_phrase + ", " + enhanced_phrases_str if enhanced_phrases else edited_phrase
                prompts.append(edited_phrase + ". " + data["edit_info"]["edited_caption"])
                strenth = 0.99

            REPEAT = 2
            generated_images_batch = []
            for _ in range(REPEAT):
                num_inference_steps = (
                    random.choice(range(20, 50, 5))
                    if self.diffuser.diffusion_model_name == "sdxl"
                    else random.choice(range(35, 75, 5))
                )
                config = {
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": random.choice(np.arange(5, 12, 1.5).tolist()),
                    "strength": strenth,
                }
                use_negative_prompt = random.choice([True, False])
                generated_images = self.diffuser.sd_masked_inpainting(
                    prompt=prompts,
                    image=input_images,
                    mask_image=mask_images,
                    negative_prompt=[negative_prompt] * len(prompts) if use_negative_prompt else None,
                    **config,
                    num_images_per_prompt=num_images_per_prompt,
                )
                generated_images_batch.extend(generated_images)
            generated_images_batch = group_outputs_for_batch_repeats(
                generated_images_batch, batch_size, REPEAT, num_images_per_prompt
            )

            for data in batch_data:
                data.update(
                    generated_images=generated_images_batch.pop(0),
                    use_negative_prompt=use_negative_prompt,
                    **config,
                )
            assert len(generated_images_batch) == 0, "generated_images should be empty"

        outputs = []
        for data in image_data_list:
            generated_images = data["generated_images"]
            if not generated_images:
                continue
            random.shuffle(generated_images)
            # use vqa scorer to verify the generated images
            generated_image, result_dict = None, None
            for image in generated_images:
                qa_for_edit = self.generated_qa_annotations.get(str(image_id), {}).get(data["edit_info"]["edit_id"])
                question_answer_pairs_regcap = qa_for_edit.get("generated_questions_region_caption", [])
                try:
                    result_dict = vqa_scorer.get_tifa_score(question_answer_pairs_regcap, image, enable_logging=False)
                except Exception as e:
                    logger.error(f"Error in get_tifa_score: {e}")
                    continue
                generated_image = image
                if result_dict and result_dict.get("tifa_score") == 1:
                    break
            logger.info(f"Generated image with TIFA score: {result_dict.get('tifa_score')}")
            diffusion_params = {
                "num_inference_steps": data["num_inference_steps"],
                "use_negative_prompt": data["use_negative_prompt"],
                "guidance_scale": data["guidance_scale"],
            }
            generated_image_rsz = generated_image.resize(input_image.size)
            annotated_image, boxes_xyxy = annotate_image_with_boxes(
                generated_image_rsz, data["boxes"], [data["edit_info"]["edited_phrase"]] * len(data["boxes"])
            )
            image_generation_info = {
                "edited_image": generated_image_rsz,
                "annotated_image": annotated_image,
                "boxes": boxes_xyxy,
                "tifa_score": result_dict.get("tifa_score"),
            }

            outputs.append(
                {
                    **data["edit_info"],
                    "bounding_boxes": data["boxes"],
                    "diffusion_params": diffusion_params,
                    "image_generation_info": image_generation_info,
                }
            )
        end_time = time.time()
        total_time_taken = end_time - start_time
        logger.info(f"Total time taken to generate {len(outputs)} images: {total_time_taken:.2f} seconds")

        return outputs

    def run(self, annotations: dict, output_dir_root: str, debug=False):
        """
        coco_captions: a list of coco captions
        diffuser: a diffuser object
        """
        logger.info(f"Total entries available for processing: {len(annotations)}")
        annotations = [
            entry
            for entry in annotations
            if self.generated_edit_instructions.get(str(entry["image_id"]), {}).get(entry["caption"])
        ]
        logger.info(f"Total entries available for processing after filtering: {len(annotations)}")

        tot_processed, success_count = 0, 0
        for idx, entry in tqdm(enumerate(annotations), desc="Editing images"):
            image_id = entry["image_id"]
            caption_text = entry["caption"]  # let's treat caption as prompt for stable-diffuser
            if "image_path" in entry:
                image_path = entry["image_path"]
            else:
                coco_split = "val" if self.split == "validation" else self.split
                image_path = get_coco_path_by_image_id(split=coco_split, image_id=image_id)
            logger.info(f"Processing image id: {image_id}, image caption: {caption_text}")
            output_dir = self.get_output_dir_path_by_image_id(output_dir_root, image_id)

            # if self.dataset == "coco" and len([f for f in os.listdir(output_dir) if f.endswith(".png")]) >= 2:
            #     logger.info(f"Skipping image id: {image_id} as 2 png files already exist.")
            #     continue

            annotation_file_path = os.path.join(output_dir, "annotations.json")

            with self.file_locker.locked(output_dir) as lock_acquired:
                if not lock_acquired:
                    logger.warning(f"Skipping image id: {image_id} as another process is working on it.")
                    continue

                edits_info = self.get_edit_instruction(image_id, caption_text, annotation_file_path)
                if not edits_info:
                    continue

                logger.info(f"Processing IDX # {idx}")
                logger.info(f"Original: {caption_text}, Edited: {edits_info}")

                edited_image_list = self.image_diffusion_edit_and_rank(image_id, image_path, caption_text, edits_info)
                if not edited_image_list:
                    logger.warning(f"No edited images generated for image id: {image_id}")
                    continue

                saved_paths = []
                annotations_dict = {"input_caption": caption_text, "input_image_path": image_path}
                for info in edited_image_list:
                    curr_gen_info = info.pop("image_generation_info")
                    edited_image = curr_gen_info["edited_image"]
                    edited_image_path = self.get_image_save_path(output_dir, info["edit_id"])
                    edited_image.save(edited_image_path)
                    if debug:
                        annotated_image = curr_gen_info["annotated_image"]
                        annotated_image_path = os.path.splitext(edited_image_path)[0] + "_annotated.png"
                        annotated_image.save(annotated_image_path)
                    saved_paths.append(edited_image_path)
                    info.update({"scores": {edited_image_path: {"tifa_score": curr_gen_info["tifa_score"]}}})
                    success_count += curr_gen_info.get("tifa_score", 0) == 1
                    tot_processed += 1

                annotations_dict.update({"annotations": edited_image_list})
                if saved_paths:
                    self.save_generated_image_annotations(annotations_dict, annotation_file_path)
                    formatted_saved_paths = "\n".join(f"- {item}" for item in saved_paths)
                    logger.info(
                        f"Total image generated: {len(edited_image_list)}. Saved generated images at:\n{formatted_saved_paths}"
                    )
                    logger.info(f"Perc. success: {success_count / tot_processed * 100:.2f}%")

    def get_output_dir_path_by_image_id(self, output_dir: str, image_id):
        base_path = os.path.join(output_dir, f"{image_id}")
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        return base_path

    @staticmethod
    def save_generated_image_annotations(annotations_dict: dict, output_file_path: str):
        """
        annotations_dict: a dictionary of image paths and corresponding captions
        """
        if len(annotations_dict) == 0:
            return

        if os.path.exists(output_file_path):
            with open(output_file_path, "r") as f:
                existing_annotations_dict = json.load(f)
            # Merge the new annotations with the existing ones on the "annotations" key
            existing_annotations_dict["annotations"].extend(annotations_dict["annotations"])
            annotations_dict = existing_annotations_dict  # Use the merged annotations

        with open(output_file_path, "w") as f:
            json.dump(annotations_dict, f, cls=CustomJsonEncoder, indent=4)

        logger.info(f"Saved annotations at {output_file_path}")

    @staticmethod
    def get_image_save_path(output_dir: str, edit_id: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{edit_id}.png")
        return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "flickr30k", "vsr"],
        help="Dataset to use. Possible values: coco, flickr30k.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use. Possible values: train, validation, test.",
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        default=None,
        help="Index of the chunk to process. This is used to split the dataset into smaller chunks for parallel processing.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="edit_suggestion",
        help=f"Type of prompt to use. Possible values: instruct_pix2pix. Each choice represents a different strategy for generating prompts.",
    )
    parser.add_argument(
        "--diffusion_model_name",
        type=str,
        default="sdxl",
        choices=["sd2", "sdxl"],
        help=f"Type of edit engine to use. Possible values: gligen, sdxl. Each choice represents a different strategy for generating edits.",
    )
    parser.add_argument(
        "--language_model_name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        choices=LANGUAGE_MODEL_NAMES,
        help=f"Set pre-trained language model to use. Possible values: {', '.join(LANGUAGE_MODEL_NAMES)}.",
    )
    args = parser.parse_args()

    args_dict_formatted = ", ".join([f"`{attr}`: {getattr(args, attr)}" for attr in vars(args)])
    logger.info(f"Arguments: ({args_dict_formatted})")

    dataset = args.dataset
    split = args.split
    chunk_index = args.chunk_index
    prompt_type = args.prompt_type
    language_model_name = args.language_model_name  # "lmsys/vicuna-13b-v1.3" or "eachadea/vicuna-13b-1.1"
    diffusion_model_name = args.diffusion_model_name
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
    language_model_name_str = re.sub(r"[/\-]", "", language_model_name).lower()

    # load coco annotations from filepath `captions_train_2014.json`
    if args.dataset == "coco":
        if split == "train" and chunk_index is None:
            raise ValueError("Chunk index must be provided for processing the training split of COCO dataset.")
        annotations = load_coco_captions(split=split, chunk_index=chunk_index)
        random.shuffle(annotations)

    elif args.dataset == "vsr":
        annotations = load_vsr_captions(split=split)
        random.shuffle(annotations)

    logger.warning(f"Using {len(annotations)} images from COCO dataset")
    # if llm based prompt is used, cache the full dataset of imagenet captions
    cache_dir = os.path.join(
        SYNTH_DIFFUSE_DATA_DIR,
        "prompt_resources",
        f"llm_edits_{dataset}",
        language_model_name_str,
    )

    masked_inpainter = MaskedInpainter(
        ckpt_repo_id, ckpt_filename, ckpt_config_filename, diffusion_model_name=diffusion_model_name, device=device
    )
    model_name_or_path = "llava-hf/llava-v1.6-mistral-7b-hf"
    vqa_scorer = VQAModelForTifa(model_name_or_path, load_in_Nbit=8)
    image_edit_proc = ImageEditProcessor(
        diffuser=masked_inpainter,
        vqa_scorer=vqa_scorer,
        prompt_type=prompt_type,
        cache_dir=cache_dir,
        split=split,
        dataset=dataset,
        chunk_index=chunk_index,
        device=device,
    )
    output_dir_root = os.path.join(VIM_DATA_DIR, f"{dataset}_{diffusion_model_name}_edited_{split}")
    image_edit_proc.run(annotations, output_dir_root)
