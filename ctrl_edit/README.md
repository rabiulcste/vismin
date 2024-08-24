# Controllable Image-Text Editing

## Installation

Before diving in, ensure you've installed the required dependencies: [PyTorch](https://pytorch.org/get-started/locally/), [Diffuser](https://github.com/openai/CLIP), [CLIP](https://github.com/openai/CLIP), and [Transformers](https://github.com/huggingface/transformers).


## Generating Minimal-Change Pairs for the Object-Attribute Category (COCO Dataset)

To produce hard negative images for the COCO dataset, follow the outlined steps:

### 1. Modify COCO Captions

To achieve visual-minimal change image-text pairs for the COCO dataset, it's essential to minimally adjust the captions. This can then pave the way for generating new images using Text-to-Image Diffusion models. Such alterations typically revolve around modifying the objects, attributes, or relations depicted in the images. 

For this procedure, we leverage the capabilities of the LLM (Language Language Model). When fed with the original COCO caption, the LLM responds by providing the necessary edit instructions, subsequently restructured caption, and supplementary metadata highlighting the nature of the change made.

For your convenience, we've curated a set of prompt templates. These are stored within the `promptsource/` directory.These templates are designed to generate edit instructions and are tailored specifically for a range of diffusion models.  For instance, the `promptsource/dino_glinen.json` file contains prompts for the DINO+GLINEN model.


To generate LLM-guided edit instructions, execute the following command, filling in the placeholders as needed:
```bash
python3 -m llm_agent.edit_instructgen_llm --prompt_type <> --language_model_name <> --dataset <> --output_dir <>
```

### 2. Generate Hard Negative Images

Depending on the model you want to utilize, run the corresponding command:

| Edit type          | Command                                               |
|----------------|-------------------------------------------------------|
| Edit existing image with mask    | `python3 -m ctrl_edit.diffusion_with_mask --args <>`           | Edit an image with a mask and a text prompt. |
| Create new image With layout   | `python3 -m ctrl_edit.diffusion_with_layout --args <>`          | Create an image with a layout and a text prompt. |
| Edit existing image for spatial change | `python3 -m ctrl_edit.inpait_with_layout_spatial --args <>`          | Edit an image with a spatial change and a text prompt. |
| Edit existing image for counting change | `python3 -m ctrl_edit.inpait_with_layout_counting --args <>`          | Edit an image with a counting change and a text prompt. |


### 3. Validate and Filter Generated Images
To ensure the quality and relevance of the edited images, we utilize a local-global region vqa filter. This filter is designed to evaluate the faithfulness of the generated images to the provided text prompts.

```bash
# to verify object, attribute, and relation changes
python3 -m filters.tifa_filter --args <> 
# to verify counting changes
python3 -m filters.counting_filter --args <> 
```


## Data Preparation
**Note:**
Fine-tuning models on this dataset can enhance their precision in discerning subtle visual discrepancies between images and their captions, leading to improved performance in tasks requiring attention to nuanced details.

**Image-text Hard-negative Pairs:** Following the previous steps, you're now ready to generate a dataset containing hard-negative pairs. These pairs introduce images with minimal alterations alongside both their original and altered captions.


To generate the training dataset for CLIP-like model fine-tuning, use the following script:
```bash
python3 -m data_maker.prepare_hn_data_clip.py --input_dir <path_to_verified_images> --output_dir <path_to_final_dataset>
```
Each entry in the dataset is structured as follows, including the original image-text pair and its corresponding hard-negative instances:

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

To generate the training dataset for Idefics2 (an MLLM) model fine-tuning, use the following script:
```bash
python3 -m data_maker.prepare_hn_data_mllm.py --input_dir <path_to_verified_images> --output_dir <path_to_final_dataset>
```

Each entry in the dataset is structured as follows, including the original image-text pair and its corresponding hard-negative instances:
```csv
type,image_path_1,image_path_2,question,answer,edit_id
multi-image,/network/projects/aishwarya_lab/datasets/coco/images/train2017/000000108314.jpg,/network/projects/mair_project_vim/coco_sdxl_edited_train/108314/4f6671cd.png,"Which image better aligns with the description: ""Two blueberry waffles topped with butter sit on a white plate.""? The first or the second image?",First,108314.4f6671cd.image_task
multi-image,/network/projects/mair_project_vim/coco_sdxl_edited_train/108314/4f6671cd.png,/network/projects/aishwarya_lab/datasets/coco/images/train2017/000000108314.jpg,"Which image better aligns with the description: ""Two blueberry waffles topped with butter sit on a white plate.""? The first or the second image?",Second,108314.4f6671cd.image_task
multi-image,/network/projects/aishwarya_lab/datasets/coco/images/train2017/000000108314.jpg,/network/projects/mair_project_vim/coco_sdxl_edited_train/108314/4f6671cd.png,"Which image better aligns with the description: ""Two pancakes topped with butter sit on a white plate.""? The first or the second image?",Second,108314.4f6671cd.image_task
multi-image,/network/projects/mair_project_vim/coco_sdxl_edited_train/108314/4f6671cd.png,/network/projects/aishwarya_lab/datasets/coco/images/train2017/000000108314.jpg,"Which image better aligns with the description: ""Two pancakes topped with butter sit on a white plate.""? The first or the second image?",First,108314.4f6671cd.image_task
image,/network/projects/aishwarya_lab/datasets/coco/images/train2017/000000108314.jpg,,"Does this image depict: (A) Two blueberry waffles topped with butter sit on a white plate., or (B) Two pancakes topped with butter sit on a white plate.?",A,108314.4f6671cd.text_task
image,/network/projects/aishwarya_lab/datasets/coco/images/train2017/000000108314.jpg,,"Does this image depict: (A) Two pancakes topped with butter sit on a white plate., or (B) Two blueberry waffles topped with butter sit on a white plate.?",B,108314.4f6671cd.text_task
image,/network/projects/mair_project_vim/coco_sdxl_edited_train/108314/4f6671cd.png,,"Does this image depict: (A) Two blueberry waffles topped with butter sit on a white plate., or (B) Two pancakes topped with butter sit on a white plate.?",B,108314.4f6671cd.text_task
image,/network/projects/mair_project_vim/coco_sdxl_edited_train/108314/4f6671cd.png,,"Does this image depict: (A) Two pancakes topped with butter sit on a white plate., or (B) Two blueberry waffles topped with butter sit on a white plate.?",A,108314.4f6671cd.text_task
```
