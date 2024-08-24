import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict

import clip
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from ctrl_edit.utils.constants import LANGUAGE_MODEL_NAMES, VIM_DATA_DIR
from ctrl_edit.utils.helpers import (format_caption, get_coco_path_by_image_id,
                                     load_coco_captions, set_random_seed)
from ctrl_edit.utils.logger import Logger

# Set up logger
logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed()


class MllmTrainDataProcessor:
    def __init__(self):
        self.cache_file_path = None
        self.cache = None

    @staticmethod
    def _load_cached_data(json_file):
        if os.path.exists(json_file):
            logger.info(f"Loading existing cache from {json_file}")
            with open(json_file, "r") as f:
                return json.load(f)

        return defaultdict(dict)

    def set_cache_file_path(self, cache_file_path):
        if not os.path.exists(os.path.dirname(cache_file_path)):
            logger.warning(f"Cache directory does not exist. Creating directory: {os.path.dirname(cache_file_path)}")
            os.makedirs(os.path.dirname(cache_file_path))
        self.cache_file_path = cache_file_path

    def parse_qa_pairs_from_string(self, input_string) -> list:
        """
        Extracts questions, choices, and answers from the provided string.

        Args:
        - input_string (str): The string containing questions, choices, and answers.

        Returns:
        - list of dict: Each dictionary contains a Question, its Choices, and the Answer.
        """
        input_string = input_string.strip("[]")

        # Regex pattern to capture each question and answer block cleanly
        pattern = re.compile(r"'Question: (.*?) Answer: (.*?)'(?:, |$)", re.DOTALL)

        # Find all matches using the regex pattern
        matches = pattern.findall(input_string)

        # Convert each match into a dictionary format
        qa_pairs = [{"user": q.strip(), "assistant": a.strip()} for q, a in matches]

        return qa_pairs

    @staticmethod
    def create_entry(type, image_path_1, image_path_2, question, answer, edit_id):
        return {
            "type": type,
            "image_path_1": image_path_1,
            "image_path_2": image_path_2,
            "question": question,
            "answer": answer,
            "edit_id": edit_id,
        }

    @staticmethod
    def generate_image_mcq_prompt(caption):
        query_templates = [
            f'You are given two images. Which one better aligns with the description: "{caption}"? The first or the second image?',
            f'Question: You are given two images. Which one better aligns with the description? "{caption}" Choices: First, Second.',
            f'Question: Two images are presented. Which one better aligns with the description: "{caption}"? Options: First, Second.',
            f'Based on the description: "{caption}", please select one of the given options that best matches the image. Choices: First, Second.',
            f'Which image corresponds more accurately to the description: "{caption}"? Image 1 or Image 2?',
            f'Based on the description: "{caption}", which image is more appropriate? Is it the first or the second image?',
            f'Considering the description: "{caption}", which image would you say matches it better? The first one or the second one?',
            f'Given the description: "{caption}", which image would you select? The first or the second?',
            f'If the description is: "{caption}", which image would you choose? The first or the second?',
            f'Which image is more likely to be described as: "{caption}"? The first or the second?',
            f'Which image fits the description: "{caption}" more accurately? The first or the second?',
            f'If you were to match the description: "{caption}" with one of the images, which one would it be? The first or the second?',
            f'Given the caption: "{caption}", which of the two images aligns more closely with this description? Choose either First or Second.',
            f'Considering the caption: "{caption}", select the image that best represents this description. Is it Image 1 or Image 2?',
            f'For the description: "{caption}", which image would you say fits the description more closely? First or Second?',
            f'Reflecting on the description: "{caption}", which of these two images would you associate with it? First or Second?',
            f'With the description: "{caption}" in mind, which image would you say is a better match? First or Second?',
            f'Analyzing the description: "{caption}", choose the image that best captures the essence of this caption. Is it the first or the second?',
            f'Based on the description given as: "{caption}", which of the two images reflects the description more precisely? First or Second?',
            f'When considering the description: "{caption}", which of the two given images more accurately represents it? Choose between the first and the second.',
            f'Taking into account the description: "{caption}", which of the two provided images would you select as the more accurate depiction? First or Second?',
            f'Look at the description: "{caption}". Of the two given images, which one do you think corresponds better to it? Option: First or Second?',
            f'From the given description: "{caption}", which of the two images do you believe more accurately captures the described scene? First or Second?',
            f'Using the description: "{caption}" as a guide, which of the two images do you feel better captures its spirit? First or Second?',
        ]
        return random.choice(query_templates)

    @staticmethod
    def get_image_task_mcq_answer(choice, description):
        answer_templates = [
            f"The correct answer is: {choice.lower()} image.",
            f"The answer is: {choice.lower()} image.",
            f"The image that better aligns with the description is the {choice.lower()} image.",
            f"The {choice.lower()} image better matches the description.",
            f'The {choice.lower()} image better aligns with the description: "{description}"',
            f"The image that corresponds more accurately to the description is the {choice.lower()} image.",
            f"The {choice.lower()} image is more appropriate.",
            f"{choice}",
            f'The image that best fits the description: "{description}" is the {choice.lower()} image.',
            f'The image that is more likely to be described as "{description}" is the {choice.lower()} image.',
            f'The image that is a better match for the description "{description}" is the {choice.lower()} image.',
        ]
        return random.choice(answer_templates)

    @staticmethod
    def get_text_task_mcq_prompt(candidate_texts):
        # remove last full stop from the candidate texts
        candidate_texts = [re.sub(r"\.$", "", text) for text in candidate_texts]
        query_templates = [
            f"Question: Does this image depict: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}? Choices: A, B.",
            f"Does this image best match: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Does this image best represent: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Does this image best describe: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}? Choices: A, B.",
            f"Question: Which of the following best matches the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}? Choices: A, B.",
            f"Question: Does this image best characterize: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Does this image best capture: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Does this image best illustrate: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best describes the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best represents the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best matches the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best captures the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best illustrates the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best describes the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best represents the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best matches the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best captures the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which of the following best illustrates the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which description is more accurate for the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which description is more fitting for the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which description is more suitable for the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which description is more appropriate for the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which description is more precise for the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which description is more exact for the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
            f"Which description is more correct for the image: (A) {candidate_texts[0]}, or (B) {candidate_texts[1]}?",
        ]
        return random.choice(query_templates)

    @staticmethod
    def get_text_task_mcq_answer(choice, caption):
        answer_templates = [
            f"The correct answer is: ({choice}) {caption}",
            f"The answer is: ({choice}) {caption}",
            f"The description that best fits the image is: ({choice}) {caption}",
            f"The ({choice}) {caption} description better matches the image.",
            f"The description that corresponds more accurately to the image is: ({choice}) {caption}.",
            f"({choice}) {caption.capitalize()} description is more appropriate.",
            f"({choice}) {caption}",
            f"The description that best matches the image is: ({choice}) {caption}.",
            f"The description that is a better match for the image is: ({choice}) {caption}.",
            f"The correct answer is: {choice}.",
            f"The answer is: {choice}.",
            f"The correct choice is: {choice}.",
            f"{choice}",
        ]
        return random.choice(answer_templates)

    def get_join_description_prompt(self, captions):
        captions = [re.sub(r"\.$", "", text) for text in captions]
        captions = [caption.lower() for caption in captions]
        answer_templates = [
            f"The first image shows: {captions[0]}. The second image shows: {captions[1]}.",
            f"First, we see an image where {captions[0]}. Then, the second image shows {captions[1]}.",
            f"In the first image, {captions[0]}. In the second image, {captions[1]}.",
            f"In the image on the left, {captions[0]}. In the image on the right, {captions[1]}.",
            f"The first image shows {captions[0]}, while the second image shows {captions[1]}.",
            f"The image on the left displays {captions[0]}, while the image on the right displays {captions[1]}.",
        ]
        return random.choice(answer_templates)

    def export_itm_data(self, training_data: str, out_fpath: str):
        IMAGE_CAPTION_TASK_SAMPLE_RATE = 0.25
        # let's save winoground task data
        itm_data = []
        for sample in training_data:
            image_id = sample["image_id"]
            sample["caption"] = format_caption(sample["caption"])
            for info in sample["negatives"]:
                edit_id = info["edit_id"]
                info["edited_caption"] = format_caption(info["edited_caption"])
                images = [sample["image_path"], info["edited_image_path"]]
                captions = [sample["caption"], info["edited_caption"]]

                # image type
                if random.random() < IMAGE_CAPTION_TASK_SAMPLE_RATE:
                    itm_data.append(self.create_entry("image", images[1], "", "", captions[1], edit_id))
                if random.random() < IMAGE_CAPTION_TASK_SAMPLE_RATE:
                    itm_data.append(self.create_entry("image", images[0], "", "", captions[0], edit_id))

                if random.random() < IMAGE_CAPTION_TASK_SAMPLE_RATE:
                    joint_caption = self.get_join_description_prompt(captions)
                    itm_data.append(
                        self.create_entry(
                            "multi-image",
                            images[0],
                            images[1],
                            "",
                            joint_caption,
                            f"{image_id}.{edit_id}.joint_description",
                        )
                    )

                for caption_idx, selected_caption in enumerate(captions):
                    # image_task_question = f'Which image better aligns with the description: "{selected_caption}"? The first or the second image?'
                    image_task_question = self.generate_image_mcq_prompt(selected_caption)
                    labels = ["First", "Second"] if caption_idx == 0 else ["Second", "First"]
                    i = random.choice([0, 1])
                    choice = labels[i]
                    image_task_answer = self.get_image_task_mcq_answer(choice, selected_caption)
                    itm_data.append(
                        self.create_entry(
                            "multi-image",
                            images[i],
                            images[1 - i],
                            image_task_question,
                            image_task_answer,
                            f"{image_id}.{edit_id}.image_task",
                        )
                    )

                for image_idx, selected_image in enumerate(images):
                    labels = ["A", "B"]
                    candidate_texts = [captions[0], captions[1]]
                    i = random.choice([0, 1])
                    candidate_texts = [candidate_texts[i], candidate_texts[1 - i]]  # A, B
                    candidate_labels = [labels[i], labels[1 - i]]  # A, B
                    description = captions[image_idx]
                    answer = candidate_labels[image_idx]

                    text_task_question = self.get_text_task_mcq_prompt(candidate_texts)
                    text_task_answer = self.get_text_task_mcq_answer(answer, description)

                    # print(text_task_question, text_task_answer)
                    itm_data.append(
                        self.create_entry(
                            "image",
                            selected_image,
                            "",
                            text_task_question,
                            text_task_answer,
                            f"{image_id}.{edit_id}.text_task",
                        )
                    )

        with open(out_fpath, "w", newline="") as f:
            fieldnames = ["type", "image_path_1", "image_path_2", "question", "answer", "edit_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for item in itm_data:
                writer.writerow(item)

        logger.info(f"Saved generative data to {out_fpath}")

    # Function to load image and apply preprocessing
    @staticmethod
    def load_and_preprocess_image(image_path, preprocess):
        try:
            image = Image.open(image_path).convert("RGB")
            return preprocess(image)
        except Exception as e:
            print(f"Error loading or processing image {image_path}: {e}")
            return None


def do_nearest_neighbor_pairing(split, sampling_strategy):
    caption_data = load_coco_captions(split)

    image_text_pairs = []
    for curr_image_data in caption_data:
        caption = curr_image_data["caption"]
        image_path = get_coco_path_by_image_id(split, curr_image_data["image_id"])
        image_text_pairs.append({"image_path": image_path, "caption": caption, "image_id": curr_image_data["image_id"]})

    # do nearest neighbor sampling using SBERT if nearest_txt
    if sampling_strategy == "nearest_txt":
        model = SentenceTransformer("paraphrase-distilroberta-base-v1")  # Load the pre-trained SBERT model
        annotations = []
        embeddings = model.encode([sample["caption"] for sample in image_text_pairs], convert_to_tensor=True)
        for index, sample in tqdm(
            enumerate(image_text_pairs), desc="Finding Similar Captions", total=len(image_text_pairs)
        ):
            query_embedding = model.encode(sample["caption"], convert_to_tensor=True)
            cos_similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]

            sorted_indices = cos_similarities.argsort(descending=True).tolist()
            sorted_indices.remove(index)

            for similar_idx in sorted_indices:
                # If all are 100% matches (very unlikely but just in case), then skip
                if cos_similarities[similar_idx].item() >= 0.8:
                    continue
                edit_id = image_text_pairs[similar_idx]["image_path"].split("/")[-1].split(".")[0]
                # print(f"{sample['caption']} => {image_text_pairs[similar_idx]['caption']}")
                annotations.append(
                    {
                        "image_id": sample["image_id"],
                        "image_path": sample["image_path"],
                        "caption": sample["caption"],
                        "negatives": [
                            {
                                "edited_image_path": image_text_pairs[similar_idx]["image_path"],
                                "edited_caption": image_text_pairs[similar_idx]["caption"],
                                "edit_id": edit_id,
                                "category": "object",
                            }
                        ],
                    }
                )
                break

    # do nearest neighbor sampling using clip if nearest_img
    elif sampling_strategy == "nearest_img":
        # Assuming model and preprocess are loaded outside the loop for efficiency
        model, preprocess = clip.load("ViT-B/32", device=device)
        annotations = []
        # Encode images individually to see progress
        new_image_text_pairs = []
        image_embeddings = []
        for sample in tqdm(image_text_pairs, desc="Encoding Images"):
            try:
                image = Image.open(sample["image_path"]).convert("RGB")
                processed_image = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model.encode_image(processed_image)
                    image_embeddings.append(embedding)
                new_image_text_pairs.append(sample)
            except Exception as e:
                print(f"Error processing image: {sample['image_path']}. Error: {e}")
        image_text_pairs = new_image_text_pairs
        # Stack all embeddings into a single tensor
        image_embeddings = torch.stack(image_embeddings).squeeze()

        for index, sample in tqdm(
            enumerate(image_text_pairs), desc="Finding Similar Images", total=len(image_text_pairs)
        ):
            query_embedding = image_embeddings[index].unsqueeze(0)  # Reuse the precomputed embedding
            cos_similarities = util.pytorch_cos_sim(query_embedding, image_embeddings)[0]

            sorted_indices = cos_similarities.argsort(descending=True).tolist()
            sorted_indices.remove(index)

            for similar_idx in sorted_indices:
                if cos_similarities[similar_idx].item() < 0.99:
                    edit_id = image_text_pairs[similar_idx]["image_path"].split("/")[-1].split(".")[0]
                    annotations.append(
                        {
                            "image_id": sample["image_id"],
                            "image_path": sample["image_path"],
                            "caption": sample["caption"],
                            "negatives": [
                                {
                                    "edited_image_path": image_text_pairs[similar_idx]["image_path"],
                                    "edited_caption": image_text_pairs[similar_idx]["caption"],
                                    "edit_id": edit_id,
                                    "category": "object",
                                }
                            ],
                        }
                    )
                    print(f"{sample['caption']} => {image_text_pairs[similar_idx]['caption']}")
                    break  # Exit after the first suitable negative is found

    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use. Possible values: train, val, test.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="visdiff",
        help=f"Type of prompt to use. Possible values: visdiff. Each choice represents a different strategy for generating prompts.",
    )
    parser.add_argument(
        "--language_model_name",
        type=str,
        default="teknium/OpenHermes-2.5-Mistral-7B",
        choices=LANGUAGE_MODEL_NAMES,
        help=f"Set pre-trained language model to use. Possible values: {', '.join(LANGUAGE_MODEL_NAMES)}.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generating edit instructions.",
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="ours",
        choices=["ours", "nearest_img", "nearest_txt", "mixed"],
        help="Sampling strategy to use for pairing image-caption pairs.",
    )

    args = parser.parse_args()

    args_dict_formatted = ", ".join([f"`{attr}`: {getattr(args, attr)}" for attr in vars(args)])
    logger.info(f"Arguments: ({args_dict_formatted})")

    prompt_type = args.prompt_type
    language_model_name = args.language_model_name  # "lmsys/vicuna-13b-v1.3" or "eachadea/vicuna-13b-1.1"
    split = args.split
    batch_size = args.batch_size
    sampling_strategy = args.sampling_strategy
    language_model_name_str = re.sub(r"[/\-]", "", language_model_name).lower()

    split_str = "training" if split == "train" else "validation"
    inp_fpath = os.path.join(
        VIM_DATA_DIR,
        "annotations",
        f"{split_str}_data",
        f"{split}.json",
    )
    out_fpath = os.path.join(VIM_DATA_DIR, "annotations", "generative", split, f"{split}_itm_{sampling_strategy}.csv")
    if sampling_strategy in ["nearest_img", "nearest_txt"]:
        annotations = do_nearest_neighbor_pairing(split, sampling_strategy)
    else:
        with open(inp_fpath, "r") as f:
            annotations = json.load(f)

    gen_proc = MllmTrainDataProcessor()
    gen_proc.export_itm_data(annotations, out_fpath)
