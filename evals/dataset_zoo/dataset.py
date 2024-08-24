import json
import os
import random
import subprocess
from pathlib import Path

import pandas as pd
import requests
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from commons.utils import format_caption

from .task_prompts import TaskPrompts

# answer ground truth
ITG_LABEL_DICT = {
    "text": ["A", "B"],
    "image": ["First", "Second"],
}
prompt_getter = TaskPrompts()


# TODO: Fix this class to getitem for both clip and mllm
class EqBenALL(Dataset):
    def __init__(self, img_root_dir, annotation_file, model_type="clip"):
        super(EqBenALL, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)

    def load_dataset(self, img_root_dir, annotation_file):
        data = json.load(open(annotation_file, "r"))

        dataset = []
        for sample in data:
            idx = "-".join(sample["image0"].split("/")[:3]).replace(
                ".png", ""
            )  # get the first 3 parts of the path to create a unique id
            split_name = sample["image0"].split("/")[0]
            sample_dict = {
                "image_0": os.path.join(img_root_dir, sample["image0"]),
                "caption_0": format_caption(sample["caption0"]),
                "image_1": os.path.join(img_root_dir, sample["image1"]),
                "caption_1": format_caption(sample["caption1"]),
                "id": idx,
                "tag": split_name,
            }
            dataset.append(sample_dict)

        return dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {
            "image_0": sample["image_0"],
            "image_1": sample["image_1"],
            "caption_0": sample["caption_0"],
            "caption_1": sample["caption_1"],
            "tag": sample["tag"],
        }

    def __len__(self):
        return len(self.dataset)


class EqBenAG(Dataset):
    def __init__(self, img_root_dir, annotation_file, config, customized_data_toolkit):
        super(EqBenAG, self).__init__()
        self.config = config
        self.sample_pair = []
        global_ann = json.load(open(annotation_file, "r"))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(global_ann.items()):
            print("Loading data {}/{} ".format(process_idx, len(self.global_ann)), end="\r")
            for subject_dir, subject_dir_values in video_dir_values.items():
                for frame_info in subject_dir_values:
                    image0_path, image1_path = os.path.join(img_root_dir, frame_info["image0"]), os.path.join(
                        img_root_dir, frame_info["image1"]
                    )
                    caption0, caption1 = (
                        frame_info["caption0"][0],
                        frame_info["caption1"][0],
                    )  # only use the first caption in the caption list
                    added_sample = {
                        "subject": subject_dir,
                        "image_0": image0_path,
                        "caption_0": caption0,
                        "image_1": image1_path,
                        "caption_1": caption1,
                    }
                    self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = (
            self.sample_pair[index]["image0"],
            self.sample_pair[index]["caption0"],
            self.sample_pair[index]["image1"],
            self.sample_pair[index]["caption1"],
        )
        image0, image1 = self.customized_data_toolkit.process_img_pixel(
            image0
        ), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, "collate"):  # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else:  # use the default pytorch collact
            return torch.utils.data.default_collate(batch)


class EqBenYoucook2(Dataset):
    def __init__(self, img_root_dir, annotation_file, config, customized_data_toolkit):
        super(EqBenYoucook2, self).__init__()
        self.config = config
        self.sample_pair = []
        global_ann = json.load(open(annotation_file, "r"))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(global_ann.items()):
            print("Loading data {}/{} ".format(process_idx, len(global_ann)), end="\r")
            for frame_info in video_dir_values:
                image0_path, image1_path = os.path.join(img_root_dir, frame_info["image0"]), os.path.join(
                    img_root_dir, frame_info["image1"]
                )
                caption0, caption1 = frame_info["caption0"], frame_info["caption1"]
                added_sample = {
                    "image0": image0_path,
                    "caption0": caption0,
                    "image1": image1_path,
                    "caption1": caption1,
                }
                self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = (
            self.sample_pair[index]["image0"],
            self.sample_pair[index]["caption0"],
            self.sample_pair[index]["image1"],
            self.sample_pair[index]["caption1"],
        )
        image0, image1 = self.customized_data_toolkit.process_img_npy(
            image0
        ), self.customized_data_toolkit.process_img_npy(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, "collate"):  # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else:  # use the default pytorch collact
            return torch.utils.data.default_collate(batch)


class EqBenGEBC(Dataset):
    def __init__(self, img_root_dir, annotation_file, config, customized_data_toolkit):
        super(EqBenGEBC, self).__init__()
        self.config = config
        self.sample_pair = []
        global_ann = json.load(open(annotation_file, "r"))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, (video_dir, video_dir_values) in enumerate(global_ann.items()):
            print("Loading data {}/{} ".format(process_idx, len(global_ann)), end="\r")
            for frame_info in video_dir_values:
                image0_path, image1_path = os.path.join(img_root_dir, frame_info["image0"]), os.path.join(
                    img_root_dir, frame_info["image1"]
                )
                caption0, caption1 = frame_info["caption0"], frame_info["caption1"]
                added_sample = {
                    "image0": image0_path,
                    "caption0": caption0,
                    "image1": image1_path,
                    "caption1": caption1,
                }
                self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = (
            self.sample_pair[index]["image0"],
            self.sample_pair[index]["caption0"],
            self.sample_pair[index]["image1"],
            self.sample_pair[index]["caption1"],
        )
        image0, image1 = self.customized_data_toolkit.process_img_pixel(
            image0
        ), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, "collate"):  # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else:  # use the default pytorch collact
            return torch.utils.data.default_collate(batch)


class EqBenKubric(Dataset):
    def __init__(self, img_root_dir, annotation_file, config, customized_data_toolkit):
        super(EqBenKubric, self).__init__()
        self.config = config
        self.sample_pair = []
        global_ann = json.load(open(annotation_file, "r"))
        self.customized_data_toolkit = customized_data_toolkit
        # import pdb; pdb.set_trace()
        for process_idx, frame_info in enumerate(global_ann):
            print("Loading data {}/{} ".format(process_idx, len(self.global_ann)), end="\r")
            image0_path, image1_path = os.path.join(img_root_dir, frame_info["image0"]), os.path.join(
                img_root_dir, frame_info["image1"]
            )
            caption0, caption1 = frame_info["caption0"], frame_info["caption1"]
            added_sample = {
                "image0": image0_path,
                "caption0": caption0,
                "image1": image1_path,
                "caption1": caption1,
            }
            self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = (
            self.sample_pair[index]["image0"],
            self.sample_pair[index]["caption0"],
            self.sample_pair[index]["image1"],
            self.sample_pair[index]["caption1"],
        )
        image0, image1 = self.customized_data_toolkit.process_img_pixel(
            image0
        ), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, "collate"):  # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else:  # use the default pytorch collact
            return torch.utils.data.default_collate(batch)


class EqBenSD(Dataset):
    def __init__(self, img_root_dir, annotation_file, config, customized_data_toolkit):
        super(EqBenSD, self).__init__()
        self.config = config
        self.sample_pair = []
        global_ann = json.load(open(annotation_file, "r"))
        self.customized_data_toolkit = customized_data_toolkit

        for process_idx, frame_info in enumerate(global_ann):
            print("Loading data {}/{} ".format(process_idx, len(global_ann)), end="\r")
            image0_path, image1_path = os.path.join(img_root_dir, frame_info["image0"]), os.path.join(
                img_root_dir, frame_info["image1"]
            )
            caption0, caption1 = frame_info["caption0"], frame_info["caption1"]
            added_sample = {
                "image0": image0_path,
                "caption0": caption0,
                "image1": image1_path,
                "caption1": caption1,
            }
            self.sample_pair.append(added_sample)

    def __getitem__(self, index):
        image0, caption0, image1, caption1 = (
            self.sample_pair[index]["image0"],
            self.sample_pair[index]["caption0"],
            self.sample_pair[index]["image1"],
            self.sample_pair[index]["caption1"],
        )
        image0, image1 = self.customized_data_toolkit.process_img_pixel(
            image0
        ), self.customized_data_toolkit.process_img_pixel(image1)
        return image0, image1, caption0, caption1

    def __len__(self):
        return len(self.sample_pair)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, "collate"):  # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else:  # use the default pytorch collact
            return torch.utils.data.default_collate(batch)


####### Conventional Benchmarks #######
class Winoground(Dataset):
    def __init__(self, img_root_dir=None, annotation_file=None, model_type="clip"):
        super(Winoground, self).__init__()
        self.model_type = model_type
        self.dataset = load_dataset("facebook/winoground")["test"]
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def get_item_clip(self, index):
        sample = self.dataset[index]
        image0, image1 = sample["image_0"].convert("RGB"), sample["image_1"].convert("RGB")
        caption0, caption1 = sample["caption_0"], sample["caption_1"]
        return {"image_0": image0, "image_1": image1, "caption_0": caption0, "caption_1": caption1}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        item_all_combinations = []
        captions = [sample["caption_0"], sample["caption_1"]]
        images = [sample["image_0"].convert("RGB"), sample["image_1"].convert("RGB")]
        text_task_prompt = prompt_getter.get_text_task_prompt(captions)
        for i in range(2):
            image_task_prompt = prompt_getter.get_image_task_prompt(captions[i])
            item_all_combinations.append(
                {
                    "type": "text",
                    "text": text_task_prompt,
                    "image": images[i],
                    "label": ITG_LABEL_DICT["text"][i],
                }
            )
            item_all_combinations.append(
                {
                    "type": "image",
                    "text": image_task_prompt,
                    "image": images,
                    "label": ITG_LABEL_DICT["image"][i],
                }
            )
        return item_all_combinations

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)

    def collate(self, batch):
        if hasattr(self.customized_data_toolkit, "collate"):  # use customized collact
            return self.customized_data_toolkit.collate(batch)
        else:  # use the default pytorch collact
            return torch.utils.data.default_collate(batch)


# Note: This class for MLLM implements limits to only 2 captions and 2 images
class SpecImage2Text(Dataset):
    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str = "clip"):
        super(SpecImage2Text, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir, annotation_file):
        with open(annotation_file, "r") as f:
            data = json.load(f)

        dataset = []
        for idx, sample in tqdm(enumerate(data), total=len(data)):
            image_path = os.path.join(img_root_dir, sample["query"])
            with Image.open(image_path) as img:
                query_image = img.convert("RGB")
            correct_idx = sample["label"]
            captions = [sample["keys"][correct_idx]] + [
                sample["keys"][i] for i in range(len(sample["keys"])) if i != correct_idx
            ]
            dataset.append(
                {
                    "image": query_image,
                    "captions": captions,  # the first caption is the correct one
                    "id": f"speci2t-{idx}",
                }
            )
        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        return {
            "image": sample["image"],
            "caption": sample["captions"],
        }

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        # we randomly choose the order of the images to avoid any bias in the model
        captions = sample["captions"][:2]
        captions = captions if index % 2 == 0 else captions[::-1]
        label = "A" if index % 2 == 0 else "B"
        prompt = prompt_getter.get_text_task_prompt(captions)
        return {
            "image": sample["image"],
            "text": prompt,
            "label": label,
        }

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)

    def collate(self, batch):
        return {
            "image": [item["image"] for item in batch],
            "caption": [item["caption"] for item in batch],
        }


# Note: This class for MLLM implements limits to only 2 captions and 2 images
class SpecText2Image(Dataset):
    def __init__(self, img_root_dir, annotation_file, model_type="clip"):
        super(SpecText2Image, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir, annotation_file):
        with open(annotation_file, "r") as f:
            data = json.load(f)

        dataset = []
        for idx, sample in tqdm(enumerate(data), total=len(data), desc="Loading SpecText2Image"):
            caption = sample["query"]
            correct_idx = sample["label"]
            images = [sample["keys"][correct_idx]] + [
                sample["keys"][i] for i in range(len(sample["keys"])) if i != correct_idx
            ]
            images = [os.path.join(img_root_dir, key) for key in images]
            dataset.append(
                {
                    "images": images,  # the first image is the correct one
                    "caption": caption,
                    "id": f"spect2i-{idx}",
                }
            )

        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        return {
            "image": sample["images"],
            "caption": sample["caption"],
        }

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        images = sample["images"][:2]
        # we randomly choose the order of the images to avoid any bias in the model
        images = images if index % 2 == 0 else images[::-1]
        label = "First" if index % 2 == 0 else "Second"
        prompt = prompt_getter.get_image_task_prompt(sample["sentence"])
        return {
            "image": images,
            "text": prompt,
            "label": label,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.get_item(index)

    def collate(self, batch):
        return {
            "images": [item["images"] for item in batch],
            "caption": [item["caption"] for item in batch],
        }


# TODO: fix this class to getitem for both clip and mllm
class VALSEv2(Dataset):
    def __init__(self, img_root_dir, annotation_file, model_type="clip"):
        super(VALSEv2, self).__init__()
        self.model_type = model_type
        self.images_path = img_root_dir
        self.foils_path = annotation_file
        self.foils_data = list(self.read_foils(self.foils_path).items())
        self.dataset = self.load_dataset()

    def load_dataset(self):
        dataset = []
        itr = 0
        for foil_id, foil in tqdm(self.foils_data):
            if "val" in foil["image_file"] and "train" in self.images_path:
                image_fpath = os.path.join(self.images_path.replace("train", "val"), foil["image_file"])
            elif "train" in foil["image_file"] and "val" in self.images_path:
                image_fpath = os.path.join(self.images_path.replace("val", "train"), foil["image_file"])
            else:
                image_fpath = os.path.join(self.images_path, foil["image_file"])
            image = Image.open(image_fpath).convert("RGB")

            captions = [foil["caption"], foil["foil"]]
            if itr % 2 == 0:
                label = "A"
            else:
                label = "B"
                captions = captions[::-1]

            item = {
                "image": image,
                "captions": captions,  # for generative
                "label": label,  # for generative
                "caption": foil["caption"],
                "foils": [foil["foil"]],
                "id": foil_id,
            }
            dataset.append(item)
            itr += 1
        return dataset

    def __getitem__(self, index):
        foil_id, foil = self.foils_data[index]

        if "val" in foil["image_file"] and "train" in self.images_path:
            image_fpath = os.path.join(self.images_path.replace("train", "val"), foil["image_file"])
        elif "train" in foil["image_file"] and "val" in self.images_path:
            image_fpath = os.path.join(self.images_path.replace("val", "train"), foil["image_file"])
        else:
            image_fpath = os.path.join(self.images_path, foil["image_file"])
        image = Image.open(image_fpath).convert("RGB")
        image = self.transform.process_img_pixel(image)
        item = {"image": image, "caption": foil["caption"], "foil": foil["foil"], "foil_id": foil_id}
        return item

    def __len__(self):
        return len(self.foils_data)

    def read_foils(self, foils_path):
        if "original-foil-dataset" in foils_path:
            foils_data = self.read_foil_dataset(foils_path)
        else:
            with open(foils_path) as json_file:
                foils_data = json.load(json_file)
        foils_data = {foil_id: foil for foil_id, foil in foils_data.items() if foil["mturk"]["caption"] > 2}
        return foils_data

    def read_foil_dataset(self, foils_path):
        """
        Read in the data of the original foil dataset and convert it on the fly to our format (dict/json).
        """
        with open(foils_path) as json_file:
            foil_dataset = json.load(json_file)

        foils_data = {}  # our format

        for foil in foil_dataset["annotations"]:
            # For unimodal models, we always need foil, non-foil pairs to compare perplexity.
            if foil["foil"] == True:  # we have a foil not foil pair
                # recover the original sentence
                orig_sentence = foil["caption"].replace(foil["foil_word"], foil["target_word"])
                image_id = foil["image_id"]
                foils_data[foil["foil_id"]] = {
                    "dataset": "FOIL dataset",
                    "dataset_idx": foil["foil_id"],
                    "original_split": "test",
                    "linguistic_phenomena": "noun phrases",
                    "image_file": f"COCO_val2014_{str(image_id).zfill(12)}.jpg",  # COCO_val2014_000000522703.jpg all are "val"
                    "caption": orig_sentence,
                    "foils": [foil["caption"]],
                    "classes": foil["target_word"],
                    "classes_foil": foil["foil_word"],
                }

        return foils_data

    def collate(self, batch):
        return torch.utils.data.default_collate(batch)


class CountBench(Dataset):
    DIGIT_TO_WORD = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }

    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str = "clip"):
        super(CountBench, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir, annotation_file):
        with open(annotation_file, "r") as f:
            data = json.load(f)

        dataset = []
        for idx, sample in tqdm(enumerate(data), total=len(data)):
            # TODO: what happened to image file download from the url?
            image_url, caption, number = sample["image_url"], sample["text"], sample["number"]
            try:
                filename = os.path.basename(image_url)
                if not filename.endswith((".jpg", ".jpeg", ".png")):
                    continue
                image_path = os.path.join(img_root_dir, filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)

                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                # print(f"Error loading image {image_url}: {e}")
                continue

            negative_captions = [
                caption.replace(self.DIGIT_TO_WORD[number], self.DIGIT_TO_WORD[i]) for i in range(2, 10) if i != number
            ]
            random.shuffle(negative_captions)
            captions = [caption, negative_captions[0]]
            dataset.append(
                {
                    "image": image,
                    "captions": captions,  # the first caption is the correct one
                    "id": f"count-{idx}",
                }
            )

        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        return {"image": sample["image"], "caption": sample["captions"]}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        # we randomly choose the order of the images to avoid any bias in the model
        captions = sample["captions"]
        captions = captions if index % 2 == 0 else captions[::-1]
        label = "A" if index % 2 == 0 else "B"
        prompt = prompt_getter.get_text_task_prompt(captions)
        return {
            "image": sample["image"],
            "text": prompt,
            "label": label,
        }

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)


class VisualSpatialRelation:
    NEGATE = {
        # Adjacency
        "adjacent to": "nonadjacent to",
        "alongside": "away from",
        "at the side of": "away from",
        "at the right side of": "at the left side of",
        "at the left side of": "at the right side of",
        "attached to": "disconnect from",
        "at the back of": "at the front of",
        "ahead of": "not ahead of",
        "against": "away from",
        "at the edge of": "far from the edge of",
        # Directional
        "off": "on",
        "past": "before",
        "toward": "away from",
        "down": "up",
        "away from": "not away from",
        "along": "not along",
        "around": "not around",
        "into": "not into",
        "across": "not accross",
        "across from": "not across from",
        "down from": "up from",
        # Orientation
        "facing": "facing away from",
        "facing away from": "facing",
        "parallel to": "perpendicular to",
        "perpendicular to": "parallel to",
        # Proximity
        "by": "far away from",
        "close to": "far from",
        "near": "far from",
        "far from": "close to",
        "far away from": "by",
        # Topological
        "connected to": "detached from",
        "detached from": "connected to",
        "has as a part": "does not have a part",
        "part of": "not part of",
        "contains": "does not contain",
        "within": "outside of",
        "at": "not at",
        "on": "not on",
        "in": "not in",
        "with": "not with",
        "surrounding": "not surrounding",
        "among": "not among",
        "consists of": "does not consists of",
        "out of": "not out of",
        "between": "not between",
        "inside": "outside",
        "outside": "inside",
        "touching": "not touching",
        # Unallocated
        "beyond": "inside",
        "next to": "far from",
        "opposite to": "not opposite to",
        "enclosed by": "not enclosed by",
        # missing
        "above": "below",
        "below": "above",
        "behind": "in front of",
        "on top of": "not on top of",
        "under": "over",
        "over": "under",
        "left of": "right of",
        "right of": "left of",
        "in front of": "behind",
        "beneath": "not beneath",
        "beside": "not beside",
        "in the middle of": "not in the middle of",
        "congruent": "incongruent",
    }

    def __init__(self, img_root_dir: str, annotation_file: str, model_type="clip"):
        super(VisualSpatialRelation, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir: str, annotation_file: str = None, split="validation"):
        data = load_dataset("cambridgeltl/vsr_random")[split]

        dataset = []
        for idx, sample in tqdm(enumerate(data), total=len(data)):
            if not sample["label"]:
                continue
            img_path = os.path.join(img_root_dir, sample["image"])
            if "val" in sample["image_link"]:
                img_path = img_path.replace("train", "val")

            caption, relation = sample["caption"], sample["relation"]
            negative_relation = self.NEGATE[relation]
            negative_caption = caption.replace(relation, negative_relation)
            captions = [caption, negative_caption]

            dataset.append(
                {
                    "image": img_path,
                    "captions": captions,  # the first index is the correct caption
                    "id": f"{split}-{idx}",
                }
            )
        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        return {"image": sample["image"], "caption": sample["captions"]}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        # we randomly choose the order of the images to avoid any bias in the model
        captions = sample["captions"]
        captions = captions if index % 2 == 0 else captions[::-1]
        label = "A" if index % 2 == 0 else "B"
        prompt = prompt_getter.get_text_task_prompt(captions)
        return {
            "image": sample["image"],
            "text": prompt,
            "label": label,
            "id": sample["id"],
        }

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)

    def collate_clip(self, batch):
        return {
            "image": [item["image"] for item in batch],
            "captions": [item["caption"] for item in batch],
        }

    def collate_mllm(self, batch):
        return {
            "image": [item["image"] for item in batch],
            "text": [item["text"] for item in batch],
            "label": [item["label"] for item in batch],
            "id": [item["id"] for item in batch],
        }

    def collate(self, batch):
        return self.collate_clip(batch) if self.model_type == "clip" else self.collate_mllm(batch)


class VisMin(Dataset):
    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str):
        super(VisMin, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if self.model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir: str, annotation_file: str):
        annotation_file = os.path.expanduser(annotation_file)
        img_root_dir = os.path.expanduser(img_root_dir)
        print(f"Loading annotations from {annotation_file} and images from {img_root_dir}")
        data = pd.read_csv(annotation_file)
        data["image_path"] = data["image_path"].apply(
            lambda x: os.path.join(img_root_dir, "original_images", x.lstrip("/"))
        )
        data["edited_image_path"] = data["edited_image_path"].apply(
            lambda x: os.path.join(img_root_dir, "edited_images", x.lstrip("/"))
        )
        return [
            {
                "image_0": row["image_path"],
                "caption_0": row["caption"],
                "image_1": row["edited_image_path"],
                "caption_1": row["edited_caption"],
                "id": row["edit_id"],
            }
            for index, row in data.iterrows()
        ]

    def get_item_clip(self, index):
        sample = self.dataset[index]
        image0, image1 = sample["image_0"], sample["image_1"]
        caption0, caption1 = sample["caption_0"], sample["caption_1"]
        return {"image_0": image0, "image_1": image1, "caption_0": caption0, "caption_1": caption1}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        captions = [sample["caption_0"], sample["caption_1"]]
        images = [sample["image_0"], sample["image_1"]]
        text_task_prompt = prompt_getter.get_text_task_prompt(captions)

        item_all_combinations = []
        for i in range(2):
            image_task_prompt = prompt_getter.get_image_task_prompt(captions[i])
            item_all_combinations.append(
                {
                    "type": "text",
                    "text": text_task_prompt,
                    "image": images[i],
                    "label": ITG_LABEL_DICT["text"][i],
                }
            )
            item_all_combinations.append(
                {
                    "type": "image",
                    "text": image_task_prompt,
                    "image": images,
                    "label": ITG_LABEL_DICT["image"][i],
                }
            )
        return {"items": item_all_combinations, "id": sample["id"]}

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)

    def collate_clip(self, batch):
        return torch.utils.data.default_collate(batch)

    def collate_mllm(self, batch):
        all_combinations = []
        ids = []
        for item in batch:
            all_combinations.append(item["items"])  # Append each item's combinations as an inner list
            ids.append(item["id"])  # Collect all ids

        return {
            "items": all_combinations,  # This should now be a 2D list
            "id": ids,  # This is a list of ids corresponding to each original sample
        }

    def collate(self, batch):
        return self.collate_clip(batch) if self.model_type == "clip" else self.collate_mllm(batch)


# TODO: Fix this class (to getitem for both CLIP and MLLM)
class MMVP(Dataset):
    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str):
        super(MMVP, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir: str, annotation_file: str):
        image_dir = os.path.join(annotation_file, "MLLM_VLM Images")
        csv_file = os.path.join(annotation_file, "Questions.csv")
        data = pd.read_csv(csv_file)
        print(f"Successfully loaded annotations from {csv_file} and images from {image_dir}")

        dataset = []
        for i in range(0, len(data), 2):
            row1 = data.iloc[i]
            row2 = data.iloc[i + 1] if i + 1 < len(data) else None

            if row2 is None:
                break

            qid1, qtype1, statement1 = row1
            qid2, qtype2, statement2 = row2
            qid1, qid2 = int(qid1), int(qid2)

            img1_path = os.path.join(image_dir, qtype1, f"{qid1}.jpg")
            img2_path = os.path.join(image_dir, qtype2, f"{qid2}.jpg")
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # uppercase the first letter of the first statement and add a period at the end if it doesn't exist
            text1 = format_caption(statement1)
            text2 = format_caption(statement2)

            dataset.append(
                {
                    "image_0": img1,
                    "image_1": img2,
                    "caption_0": text1,
                    "caption_1": text2,
                    "id": qid1,
                }
            )

        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        image0, caption0, image1, caption1 = (
            sample["image_0"],
            sample["caption_0"],
            sample["image_1"],
            sample["caption_1"],
        )

        return {"image_0": image0, "image_1": image1, "caption_0": caption0, "caption_1": caption1}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        captions = [sample["caption_0"], sample["caption_1"]]
        images = [sample["image_0"], sample["image_1"]]
        text_task_prompt = prompt_getter.get_text_task_prompt(captions)

        item_all_combinations = []
        for i in range(2):
            image_task_prompt = prompt_getter.get_image_task_prompt(captions[i])
            item_all_combinations.append(
                {
                    "type": "text",
                    "text": text_task_prompt,
                    "image": images[i],
                    "label": ITG_LABEL_DICT["text"][i],
                }
            )
            item_all_combinations.append(
                {
                    "type": "image",
                    "text": image_task_prompt,
                    "image": images,
                    "label": ITG_LABEL_DICT["image"][i],
                }
            )
        return {"items": item_all_combinations, "id": sample["id"]}

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)


class SugarCrepe(Dataset):
    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str = "clip"):
        super(SugarCrepe, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir, annotation_file):
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = []
        for idx, data in tqdm(data.items()):
            image_path = os.path.join(img_root_dir, data["filename"])
            image = Image.open(image_path).convert("RGB")
            captions = [data["caption"], data["negative_caption"]]
            dataset.append(
                {
                    "image": image,
                    "captions": captions,  # the first caption is the correct one
                    "id": idx,
                }
            )

        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        return {"image": sample["image"], "caption": sample["captions"]}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        # we randomly choose the order of the images to avoid any bias in the model
        captions = sample["captions"]
        captions = captions if index % 2 == 0 else captions[::-1]
        label = "A" if index % 2 == 0 else "B"
        prompt = prompt_getter.get_text_task_prompt(captions)
        return {
            "image": sample["image"],
            "text": prompt,
            "label": label,
        }

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)


class SpatialQA(Dataset):
    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str = "clip"):
        super(SpatialQA, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir, annotation_file):
        subset = "one" if "one" in annotation_file else "two"
        dataset_name = "vg" if "vg" in annotation_file else "coco"
        if dataset_name == "coco":
            tag_mapping = {
                "left of": "left",
                "right of": "right",
                "above": "above",
            }
        else:
            tag_mapping = {"left of": "left", "right of": "right", "front of": "front", "behind": "behind"}

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = []
        for idx, sample in tqdm(enumerate(data), total=len(data)):
            if dataset_name == "coco":
                image_path = os.path.join(img_root_dir, "{}.jpg".format(str(sample[0]).zfill(12)))
            else:
                image_path = os.path.join(img_root_dir, f"{sample[0]}.jpg")
            image = Image.open(image_path).convert("RGB")

            tag = "below" if dataset_name == "coco" else "top"  # default value
            if subset == "one":
                tag = sample[1].split()[-1]
            else:
                for phrase, mapped_tag in tag_mapping.items():
                    if phrase in sample[1]:
                        tag = mapped_tag
                        break
            captions = [sample[1], sample[2]]
            captions = [format_caption(caption) for caption in captions]
            dataset.append(
                {
                    "image": image,
                    "captions": captions,  # the first caption is the correct one
                    "tag": tag,
                    "id": idx,
                }
            )

        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        return {"image": sample["image"], "caption": sample["captions"], "tag": sample["tag"]}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        # we randomly choose the order of the images to avoid any bias in the model
        captions = sample["captions"]
        captions = captions if index % 2 == 0 else captions[::-1]
        label = "A" if index % 2 == 0 else "B"
        prompt = prompt_getter.get_text_task_prompt(captions)
        return {
            "image": sample["image"],
            "text": prompt,
            "label": label,
            "tag": sample["tag"],
        }

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)

    # TODO: based on the dataset name, the function should be used to download the dataset
    def download(self, subset="one", download=True):
        ann_root_dir = os.path.dirname(self.annotation_file)
        if subset == "one":
            annotation_file = os.path.join(ann_root_dir, "coco_qa_one_obj.json")
            image_dir = os.path.join(self.img_root_dir, "val2017")
        else:
            annotation_file = os.path.join(ann_root_dir, "coco_qa_two_obj.json")
            image_dir = os.path.join(self.img_root_dir, "val2017")

        if not os.path.exists(image_dir):
            print("Image directory for COCO-QA could not be found!")
            if download:
                os.makedirs(self.root_dir, exist_ok=True)
                image_zip_file = os.path.join(self.root_dir, "val2017.zip")
                subprocess.call(
                    ["gdown", "--no-cookies", "1zp5vBRRM4_nSik6o9PeVspDvOsHgPT4l", "--output", image_zip_file]
                )
                subprocess.call(["unzip", "val2017.zip"], cwd=self.root_dir)
            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )

        if not os.path.exists(annotation_file):
            if subset == "one":
                subprocess.call(["gdown", "--id", "1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq", "--output", annotation_file])
            else:
                subprocess.call(["gdown", "--id", "1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s", "--output", annotation_file])

        ann_root_dir = os.path.dirname(self.annotation_file)
        if subset == "one":
            annotation_file = os.path.join(ann_root_dir, "vg_qa_one_obj.json")
            image_dir = os.path.join(self.img_root_dir, "vg_images")
        else:
            annotation_file = os.path.join(ann_root_dir, "vg_qa_two_obj.json")
            image_dir = os.path.join(self.img_root_dir, "vg_images")

        if not os.path.exists(annotation_file):
            if subset == "one":
                subprocess.call(["gdown", "--id", "1ARMRzRdohs9QTr1gpIfzyUzvW20wYp_p", "--output", annotation_file])
            else:
                subprocess.call(["gdown", "--id", "1sjVG5O3QMY8s118k7kQM8zzDZH12i_95", "--output", annotation_file])

        if not os.path.exists(image_dir):
            print("Image directory for VG-QA could not be found!")
            if download:
                os.makedirs(self.root_dir, exist_ok=True)
                image_zip_file = os.path.join(self.img_root_dir, "vg_images.tar.gz")
                subprocess.call(
                    ["gdown", "--no-cookies", "1idW7Buoz7fQm4-670n-oERw9U-2JLJvE", "--output", image_zip_file]
                )
                subprocess.call(["tar", "-xvf", "vg_images.tar.gz"], cwd=self.root_dir)

            else:
                raise RuntimeError(
                    "Please either download the dataset by letting `--download` or specify the correct directory."
                )


class ImageCode(Dataset):
    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str = "clip"):
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir: str, annotation_file: str):
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = []
        for idx, sample in tqdm(enumerate(data), total=len(data)):
            img_dir = sample["directory"]
            img_files = sorted(
                (Path(img_root_dir) / img_dir).glob("*.jpg"),
                key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:]),
            )
            caption = sample["caption"].strip()
            dataset.append(
                {
                    "images": [str(img_files[sample["pos_idx"]]), str(img_files[sample["neg_idx"]])],
                    "sentence": caption,
                    "id": f"{sample['directory']}-{idx}",
                }
            )
        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        return {"image": sample["images"], "caption": sample["sentence"]}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        # we randomly choose the order of the images to avoid any bias in the model
        images = images if index % 2 == 0 else images[::-1]
        label = "First" if index % 2 == 0 else "Second"
        prompt = prompt_getter.get_image_task_prompt(sample["sentence"])
        return {
            "image": images,
            "text": prompt,
            "label": label,
        }

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)


# Note: This implementation is different from the original authors' implementation
# https://github.com/amitakamath/whatsup_vlms/blob/main/dataset_zoo/aro_datasets.py
class Whatsup(Dataset):
    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str = "clip"):
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir: str, annotation_file: str):
        with open(annotation_file, "r") as f:
            data = json.load(f)

        prepositions_mapping = {
            "left_of": "left",
            "right_of": "right",
            "_on_": "on",
            "under": "under",
            "_in-front_of_": "front",
            "_behind_": "behind",
        }
        oppposite_prepositions = {
            "left": "right",
            "right": "left",
            "on": "under",
            "under": "on",
            "front": "behind",
            "behind": "front",
        }
        dataset = []
        for idx, sample in enumerate(tqdm(data)):
            image_fname = os.path.basename(sample["image_path"])
            caption_options = sample["caption_options"]

            # Find the preposition in the image path
            preposition = next(
                (prepositions_mapping[prep] for prep in prepositions_mapping if prep in image_fname), None
            )
            query_image = Image.open(os.path.join(img_root_dir, image_fname)).convert("RGB")

            # Find the caption that contains the preposition and assign a label based on its position
            caption = None
            negative_captions = []
            for text in caption_options:
                if preposition in text.split() and caption is None:
                    caption = text
                else:
                    negative_captions.append(text)
            neg_preposition = oppposite_prepositions[preposition]
            # find the foil that contains the opposite preposition
            negative_caption = next((foil for foil in negative_captions if neg_preposition in foil.split()), None)
            captions = [caption, negative_caption]

            # Validate caption and foils
            if caption is None or preposition not in caption:
                raise ValueError(f"Caption does not contain the preposition: {caption}")
            if any(preposition in negative_caption.split() for negative_caption in negative_captions):
                raise ValueError(f"One or more foils contain the preposition: {negative_captions}")

            dataset.append(
                {
                    "image": query_image,
                    "captions": captions,
                    "id": f"whatsup-{idx}",
                }
            )

        return dataset

    def get_item_clip(self, index):
        sample = self.dataset[index]
        return {"image": sample["image"], "caption": sample["captions"]}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        # we randomly choose the order of the images to avoid any bias in the model
        captions = sample["captions"]
        captions = captions if index % 2 == 0 else captions[::-1]
        label = "A" if index % 2 == 0 else "B"
        prompt = prompt_getter.get_text_task_prompt(captions)
        return {
            "image": sample["image"],
            "text": prompt,
            "label": label,
        }

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "controlled_images.tar.gz")
        subprocess.call(["gdown", "--no-cookies", "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_images.tar.gz"], cwd=self.root_dir)
        image_zip_file = os.path.join(self.root_dir, "controlled_clevr.tar.gz")
        subprocess.call(["gdown", "--no-cookies", "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_clevr.tar.gz"], cwd=self.root_dir)
