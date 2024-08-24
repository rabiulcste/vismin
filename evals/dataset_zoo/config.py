import os
from pathlib import Path

from .dataset import (MMVP, CountBench, EqBenALL, ImageCode, SpatialQA,
                      SpecImage2Text, SpecText2Image, SugarCrepe, VALSEv2,
                      VisMin, VisualSpatialRelation, Whatsup, Winoground)

# Please identify the path of "img_root_dir" and "annotation_file" in the server.
vg_dir = "/network/projects/aishwarya_lab/datasets/vg/images/"
visual7w_dir = "/network/projects/aishwarya_lab/datasets/visual7w/images/"
coco2017val_dir = "/network/projects/aishwarya_lab/datasets/coco/images/val2017/"
coco2014val_dir = "/network/projects/aishwarya_lab/datasets/coco/images/val2014/"
coco2014train_dir = "/network/projects/aishwarya_lab/datasets/coco/images/train2014/"
coco2017train_dir = "/network/projects/aishwarya_lab/datasets/coco/images/train2017/"
swig_dir = "/network/projects/aishwarya_lab/datasets/swig/"
visDial_dir = "/network/projects/aishwarya_lab/datasets/visualdialog/VisualDialog_val2018"
coco_negatives_dir = "/network/projects/mair_project_vim/annotations/validation_data.mturk.apr28/"
count_bench_dir = "/network/projects/aishwarya_lab/datasets/countbench"
vismin_dir = "/network/projects/mair_project_vim/visMin"
sugarcrepe_annotation_file = "/network/projects/aishwarya_lab/datasets/sugar-crepe/data/"



def valse_abs_path(relative_path):
    """Resolve the absolute path given a path relative to the script."""
    script_dir = Path(__file__).resolve().parent.parent
    return str(script_dir / relative_path)

DATASET_INFO = {
    "whatsup_controlled": {
        "cls": Whatsup,
        "default": {
            "img_root_dir": "/network/projects/aishwarya_lab/datasets/whatsup/controlled_images",
            "annotation_file": "/network/projects/aishwarya_lab/datasets/whatsup/controlled_images_dataset.json",
        },
    },
    "whatsup_clevr": {
        "cls": Whatsup,
        "default": {
            "img_root_dir": "/network/projects/aishwarya_lab/datasets/whatsup/controlled_clevr",
            "annotation_file": "/network/projects/aishwarya_lab/datasets/whatsup/controlled_clevr_dataset.json",
        },
    },
    "spec_t2i": {
        "cls": SpecText2Image,
        "absolute_size": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/absolute_size",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/absolute_size/text2image.json",
        },
        "relative_size": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/relative_size",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/relative_size/text2image.json",
        },
        "absolute_spatial": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/absolute_spatial",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/absolute_spatial/text2image.json",
        },
        "relative_spatial": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/relative_spatial",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/relative_spatial/text2image.json",
        },
        "existence": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/existence",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/existence/text2image.json",
        },
        "count": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/count",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/count/text2image.json",
        },
    },
    "spec_i2t": {
        "cls": SpecImage2Text,
        "absolute_size": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/absolute_size",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/absolute_size/image2text.json",
        },
        "relative_size": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/relative_size",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/relative_size/image2text.json",
        },
        "absolute_spatial": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/absolute_spatial",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/absolute_spatial/image2text.json",
        },
        "relative_spatial": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/relative_spatial",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/relative_spatial/image2text.json",
        },
        "existence": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/existence",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/existence/image2text.json",
        },
        "count": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/count",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/spec/count/image2text.json",
        },
    },
    "imagecode": {
        "cls": ImageCode,
        "default": {
            "img_root_dir": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/image-sets",
            "annotation_file": "/home/mila/r/rabiul.awal/scratch/synth-diffuse/dataset/valid_simple.json",
        },
    },
    "gqa_spatial": {
        "cls": SpatialQA,
        "one": {
            "img_root_dir": vg_dir,
            "annotation_file": "/network/projects/aishwarya_lab/datasets/whatsup/vg_qa_one_obj.json",
        },
        "two": {
            "img_root_dir": vg_dir,
            "annotation_file": "/network/projects/aishwarya_lab/datasets/whatsup/vg_qa_two_obj.json",
        },
    },
    "coco_spatial": {
        "cls": SpatialQA,
        "one": {
            "img_root_dir": coco2017val_dir,
            "annotation_file": "/network/projects/aishwarya_lab/datasets/whatsup/coco_qa_one_obj.json",
        },
        "two": {
            "img_root_dir": coco2017val_dir,
            "annotation_file": "/network/projects/aishwarya_lab/datasets/whatsup/coco_qa_two_obj.json",
        },
    },
    "sugarcrepe": {
        "cls": SugarCrepe,
        "replace_obj": {
            "img_root_dir": coco2017val_dir,
            "annotation_file": f"{sugarcrepe_annotation_file}/replace_obj.json",
        },
        "replace_att": {
            "img_root_dir": coco2017val_dir,
            "annotation_file": f"{sugarcrepe_annotation_file}/replace_att.json",
        },
        "replace_rel": {
            "img_root_dir": coco2017val_dir,
            "annotation_file": f"{sugarcrepe_annotation_file}/replace_rel.json",
        },
        "swap_obj": {"img_root_dir": coco2017val_dir, "annotation_file": f"{sugarcrepe_annotation_file}/swap_obj.json"},
        "swap_att": {"img_root_dir": coco2017val_dir, "annotation_file": f"{sugarcrepe_annotation_file}/swap_att.json"},
        "add_obj": {"img_root_dir": coco2017val_dir, "annotation_file": f"{sugarcrepe_annotation_file}/add_obj.json"},
        "add_att": {"img_root_dir": coco2017val_dir, "annotation_file": f"{sugarcrepe_annotation_file}/add_att.json"},
    },
    "eqben_all": {
        "cls": EqBenALL,
        "default": {
            "img_root_dir": "/network/projects/aishwarya_lab/datasets/eqben/image_subset",
            "annotation_file": "/network/projects/aishwarya_lab/datasets/eqben/eqben_subset_10percent_final.json",
        },
    },
    "eqben": {
        "cls": EqBenALL,
        "default": {
            "img_root_dir": "/network/projects/aishwarya_lab/datasets/eqben/images",
            "annotation_file": "/network/projects/aishwarya_lab/datasets/eqben/all_select.json",
        },
    },
    "winoground": {
        "cls": Winoground,
        "default": {
            "img_root_dir": None,
            "annotation_file": None,
        },
    },
    "valse_v2": {
        "cls": VALSEv2,
        "existence": {"img_root_dir": visual7w_dir, "annotation_file": valse_abs_path("valse_data/existence.json")},
        "plurals": {"img_root_dir": coco2017val_dir, "annotation_file": valse_abs_path("valse_data/plurals.json")},
        "counting_hard": {
            "img_root_dir": visual7w_dir,
            "annotation_file": valse_abs_path("valse_data/counting-hard.json"),
        },
        "counting_small": {
            "img_root_dir": visual7w_dir,
            "annotation_file": valse_abs_path("valse_data/counting-small-quant.json"),
        },
        "counting_adversarial": {
            "img_root_dir": visual7w_dir,
            "annotation_file": valse_abs_path("valse_data/counting-adversarial.json"),
        },
        "relations": {
            "img_root_dir": coco2017val_dir,
            "annotation_file": valse_abs_path("valse_data/relations.json"),
        },
        "action replace": {
            "img_root_dir": swig_dir,
            "annotation_file": valse_abs_path("valse_data/action-replacement.json"),
        },
        "actant swap": {"img_root_dir": swig_dir, "annotation_file": valse_abs_path("valse_data/actant-swap.json")},
        "coref": {
            "img_root_dir": coco2014train_dir,
            "annotation_file": valse_abs_path("valse_data/coreference-standard.json"),
        },
        "coref_hard": {
            "img_root_dir": visDial_dir,
            "annotation_file": valse_abs_path("valse_data/coreference-hard.json"),
        },
        "foil_it": {"img_root_dir": coco2014val_dir, "annotation_file": valse_abs_path("valse_data/foil-it.json")},
    },
    "countbench": {
        "cls": CountBench,
        "default": {
            "img_root_dir": os.path.join(count_bench_dir, "images"),
            "annotation_file": os.path.join(count_bench_dir, "CountBench.json"),
        },
    },
    "vsr": {
        "cls": VisualSpatialRelation,
        "default": {
            "img_root_dir": coco2017train_dir,
            "annotation_file": None,
        },
    },
    "mmvp": {
        "cls": MMVP,
        "default": {"img_root_dir": None, "annotation_file": "/network/projects/aishwarya_lab/datasets/MMVP_VLM/"},
    },
    "vismin": {
        "cls": VisMin,
        "object": {"img_root_dir": vismin_dir, "annotation_file": os.path.join(vismin_dir, "object.csv")},
        "attribute": {"img_root_dir": vismin_dir, "annotation_file": os.path.join(vismin_dir, "attribute.csv")},
        "relation": {"img_root_dir": vismin_dir, "annotation_file": os.path.join(vismin_dir, "relation.csv")},
        "counting": {"img_root_dir": vismin_dir, "annotation_file": os.path.join(vismin_dir, "counting.csv")},
    },
}

# available models for openai clip
clip_model_list = [
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
]

CLIP_MODELS = {model: model.lower().replace("-", "_").replace("/", "") for model in clip_model_list}
CLIP_MODELS = {f"clip_{v}": k for k, v in CLIP_MODELS.items()}


MMVP_CATEGORIES = [
    "Orientation and Direction",
    "Presence of Specific Features",
    "State and Condition",
    "Quantity and Count",
    "Positional and Relational Context",
    "Color and Appearance",
    "Structural Characteristics",
    "Texts",
    "Viewpoint and Perspective",
]

VALSE_CATEGORIES = {
    "Existence": ["existence"],
    "Plurality": ["plurals"],
    "Counting": ["counting_hard", "counting_small", "counting_adversarial"],
    "Relations": ["relations"],
    "Actions": ["action replace", "actant swap"],
    "Coreference": ["coref", "coref_hard"],
    "Foil it": ["foil_it"],
}

POPE_CATEGORIES = ["random", "adversarial", "popular"]
SUGARCREPE_CATEGORIES = ["replace_obj", "replace_att", "replace_rel", "swap_obj", "swap_att", "add_obj", "add_att"]

import itertools

# TODO: common todos should be moved to a common place (e.g parent directory)
from commons.constants import VALID_CATEGORY_NAMES

VALSE_SUB_CATEGORIES = list(itertools.chain.from_iterable(VALSE_CATEGORIES.values()))


def get_all_splits(dataset):
    if dataset in ["coco_negatives", "vismin"]:
        all_splits = VALID_CATEGORY_NAMES  # or any other way you fetch the splits for coco_neg
    elif dataset == "valse_v2":
        all_splits = VALSE_SUB_CATEGORIES
    elif dataset == "pope":
        all_splits = POPE_CATEGORIES
    elif dataset == "sugarcrepe":
        all_splits = SUGARCREPE_CATEGORIES
    elif dataset in ["coco_spatial", "gqa_spatial"]:
        all_splits = ["one", "two"]
    elif dataset in ["spec_i2t", "spec_t2i"]:
        all_splits = ["absolute_spatial", "absolute_size", "relative_size", "relative_spatial", "existence", "count"]
    else:
        all_splits = ["default"]  # for other datasets that don't have splits

    return all_splits
