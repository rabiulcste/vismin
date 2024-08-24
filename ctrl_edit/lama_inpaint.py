import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

llama_path = str(Path(__file__).resolve().parent.parent / "lama")
if llama_path not in sys.path:
    sys.path.insert(0, llama_path)
    print(f"Adding {llama_path} to sys.path")

from lama.saicinpainting.evaluation.data import pad_tensor_to_modulo
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint


@torch.no_grad()
def inpaint_img_with_lama(img: np.ndarray, mask: np.ndarray, config_p: str, ckpt_p: str, mod=8, device="cuda"):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.0)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(predict_config.model.path, "config.yaml")

    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    checkpoint_path = os.path.join(predict_config.model.path, "models", predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location="cpu")
    model.freeze()
    if not predict_config.get("refine", False):
        model.to(device)

    batch = {}
    batch["image"] = img.permute(2, 0, 1).unsqueeze(0)
    batch["mask"] = mask[None, None]
    unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
    batch["image"] = pad_tensor_to_modulo(batch["image"], mod)
    batch["mask"] = pad_tensor_to_modulo(batch["mask"], mod)
    batch = move_to_device(batch, device)
    batch["mask"] = (batch["mask"] > 0) * 1

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    return cur_res


def build_lama_model(config_p: str, ckpt_p: str, device="cuda"):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(predict_config.model.path, "config.yaml")

    try:
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
    except FileNotFoundError:
        print(f"Error: The file '{train_config_path}' was not found.")
        print("Please download the required file using the following commands:")
        print("curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip")
        print("unzip big-lama.zip")
        # Optionally, you can exit the script if the file is critical
        exit(1)

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    checkpoint_path = os.path.join(predict_config.model.path, "models", predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    model.freeze()
    return model


@torch.no_grad()
def inpaint_img_with_builded_lama(model, img: np.ndarray, mask: np.ndarray, config_p=None, mod=8, device="cuda"):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.0)
    mask = torch.from_numpy(mask).float()

    batch = {}
    batch["image"] = img.permute(2, 0, 1).unsqueeze(0)
    batch["mask"] = mask[None, None]
    unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
    batch["image"] = pad_tensor_to_modulo(batch["image"], mod)
    batch["mask"] = pad_tensor_to_modulo(batch["mask"], mod)
    batch = move_to_device(batch, device)
    batch["mask"] = (batch["mask"] > 0) * 1

    batch = model(batch)
    cur_res = batch["inpainted"][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    return cur_res
