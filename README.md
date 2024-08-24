# VisMin: Visual Minimal-Change Understanding
**Rabiul Awal** ✨, **Saba Ahmadi** ✨, **Le Zhang** ✨, **Aishwarya Agrawal**  
*Mila - Quebec AI Institute, University of Montreal*  
✨ indicates equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2306.08832-B31B1B.svg)](https://arxiv.org/abs/2306.08832)  [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-VisMin-FFD700.svg)](https://huggingface.co/collections/mair-lab/vismin-6695660f4c450902c8aff434)


## Table of Contents
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Minimal-Change Image-Text Dataset Creation](#minimal-change-image-text-dataset-creation)
  - [LLM-guided Edit Instructions Generation](#minimal-change-text-pair-generation)
  - [Diffusion-guided Image Synthesis](#minimal-change-image-generation)
  - [Edited Image Quality Verification using Local-Global VQA Approach](#edited-image-quality-verification-using-local-global-vqa-approach)
- [Setup](#setup)
- [Acknowledgements](#acknowledgements)


## Dataset
The training dataset has 64,392 samples, and the VisMin dataset has 2,084 samples. The dataset is stored in a JSON format. Each entry contains the image path, caption, and a list of negative examples. The negative examples consist of the edited image path and edited caption.
- **Training Data:** 64,392 samples from VSR and COCO 2017 training split.
- **Benchmark Data:** 2,084 samples from COCO 2017 validation split, human-verified.


Example of a dataset entry: 
```json
{
  "image_path": "/coco/images/train2017/000000234136.jpg",
  "caption": "Two men holding a brown and white dog in a van.",
  "negatives": [
    {
      "edited_image_path": "/edited/coco/234136/0.png",
      "edited_caption": "Three men holding a brown and white dog in a van.",
    }
  ]
}
```

## Training
To fine-tune models, such as pre-trained [CLIP](), using the hard-negative contrastive loss on the curated dataset, follow these steps:

1. Clone CVPR 2024 paper's codebase: [Enhance-FineGrained](https://github.com/lezhang7/Enhance-FineGrained). 
2. You need to specify training parameters in `scrips/run_all.sh` such as `--gres=gpu:a100:2` and `batch_size`. Refer to this script file for more details.
3. To start the training, use the following commands:
```bash
cd scripts/
bash run_multiple_nodes.sh
```
The result checkpoint will be at `Enhance-FineGrained/src/Outputs` directory.

## Evaluation 
The models including CLIP or Multimodal LM can be evaluated on our VisMin benchmark which image-text matching tasks. We also support evaluation on a pool of diagnostics datasets such as VALE, Winoground, and ARO.
```bash
# To evaluate two-tower models such as CLIP
python evals.contrastive_inference --dataset <dataset_name> --model_name <path_to_model> --pretrained <pretrained_model_name>
# To evaluate generative models such as Idefics2 => https://huggingface.co/blog/idefics2
python evals.mllm_inference --dataset <dataset_name> --model_name <path_to_model>
```
## Minimal-Change Image-Text Dataset Creation
### LLM-guided Edit Instructions Generation
We use LLM to generate edit instructions. There are two approaches to generate these instructions: one with captions, which suggests object attribute changes following the style of in-context demonstrations, and another for spatial and counting changes, where we prompt LLM with in-context demonstrations to create the appropriate edit instructions with layouts.

Example of an llm-generated edit instruction (object attribute category):
```json
  {
      "InputCaption": "A glass of ice water sitting next to a wine glass.",
      "SelectedPhrase": "glass of ice water",
      "EditedPhrase": "glass of milk",
      "EditedRegionPhrase": "A glass of milk",
      "EditedCaption": "A glass of milk sitting next to a wine glass.",
      "Category": "object"
  }
```
Example of an llm-generated edit instruction (spatial and counting category):
```json
 "A paint brush is to the left of a palette.": [
      "[('a paint brush', [50, 200, 100, 312]), ('a palette', [362, 150, 150, 362])]\nBackground prompt: A realistic scene\nNegative prompt:\nCategory: relation(left of)"
  ]
```

To run the script, from the directory containing `cntr_edit/`, execute:
```bash
# for object attribute category 
# requires dataset name to be specified for source of captions
python -m llm_agent.minchange_text_pairs_gen --dataset <name_of_dataset> --prompt_type edit_instructgen_from_caption --language_model_name <name_of_language_model>
# for spatial and counting category
python -m llm_agent.minchange_text_pairs_gen --prompt_type edit_instructgen_from_caption --language_model_name <name_of_language_model>
```

Generating magic prompt (to be appended with the e.g. object name) for better diffusion guidance of input prompt:
```bash
# for object attribute category (e.g. coco dataset)
python -m llm_agent.magic_prompt --dataset coco --language_model_name <name_of_language_model>
# for spatial and counting category
python -m llm_agent.magic_prompt --dataset relation --language_model_name <name_of_language_model>
```

### Diffusion-guided Image Synthesis
We have two approaches to generate minimal-change images:
1. **Masking and Inpainting**: First, we mask the object to be edited in the source image using the [Grounding-DINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino) model. Then, we use diffusion inpainting to generate minimal-change images.
2. **Layout Swapping**: We use GLIGEN layout-diffusion to swap objects in the source image to generate edited images. For counting changes, we remove objects using their bounding boxes and create edited images.

Run the following command:
```bash
# for object attribute category (e.g. coco dataset)
# this script loads segmentation model, the diffusion model and vqa model
python -m ctrl_edit.inpaint_with_mask --language_model_name <llm_used_to_generated_edit_instruction>  --dataset <dataset_name> --output <path_to_edited_image>

# for spatial and counting category (generated from scratch using layout diffusion model)
# dataset name can be "relation" or "counting"
# this script loads the layout diffusion model and vqa model
python3 -m ctrl_edit.diffusion_llm_grounded_old --repeats 3 \
    --frozen_step_ratio 0.5 --no-scale-boxes-default \
    --sdxl --sdxl-step-ratio 0.4 \
    --dataset <dataset_name> \
    --split <split_name>
```

### Edited Image Quality Verification using Local-Global VQA Approach
Verify images through the vqa filter approach. first generate the local-global vqa questions and answers using llm following edit instruction.

```bash
# To create the local-global VQA questions and answers using LLM-generated edit instructions from one of the previous step:
python -m ctrl_edit.llm_agent.auto_filter_question_gen  --language_model_name <name_of_language_model>

# Automatically filter out bad edited images using the local-global VQA approach:
python -m ctrl_edit.filters.tifa_filter --dataset <dataset_name>
```

## Setup
```bash
git clone <https://github.com/rabiulcste/vismin>
cd vismin
pip install -r requirements.txt
```


## Acknowledgements
The codebase is built on top of the following repositories:
- [open_clip](https://github.com/mlfoundations/open_clip)
- [Enhance-FineGrained](https://github.com/lezhang7/Enhance-FineGrained)
- [GLIGEN](https://github.com/gligen/GLIGEN)
- [FastChat](https://github.com/lm-sys/FastChat)