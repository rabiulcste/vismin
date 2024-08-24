# -*- coding: utf-8 -*-
"""Idefics2 - Fine-tuning example
"""
# !pip install -q git+https://github.com/huggingface/transformers.git
# !pip install -q accelerate datasets peft bitsandbytes

"""# Loading the model and the dataset

We load the model from the Hugging Face hub. `idefics2-8b` has gone through instruction fine-tuning on a large mixture of multimodal datasets and as such is a strong starting-point to fine-tune on your own use-case. We will start from this checkpoint.

To accomodate the GPU poors, the default hyper-parameters in this tutorial are chosen so that the fine-tuning takes less than 32 GB of GPU memory. For instance, an V100 in Google Colab should be sufficient.

If you happen to have more ressources, you are encouraged to revisit some of these constraints, in particular:
- Using 4 bit quantization
- Lora fine-tuning
- Freezing the vision encoder
- Small batch size compensated with higher gradient accumulation degree
- Deactivate image splitting
- Using flash-attention
"""

import argparse
import os

import torch
from accelerate import Accelerator
from peft import LoraConfig
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, Trainer, TrainingArguments

import wandb


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(args):
    api_key = os.getenv("WANDB_API_KEY")
    if api_key is None:
        wandb.login()
    else:
        wandb.login(key=api_key)

    model_name_str = args.model_name.replace("/", "-").lower()
    model_id = f"{model_name_str}-{args.dataset_name}-{args.training_option}-lr{args.learning_rate}"
    output_dir = os.path.join(args.output_dir, model_id)
    os.makedirs(output_dir, exist_ok=True)
    try:
        import json

        wandb_json_path = os.path.join(output_dir, "wandb_resume.json")
        if os.path.exists(wandb_json_path):
            with open(wandb_json_path, "r") as f:
                run_id = json.load(f)["run_id"]
        else:
            run_id = wandb.util.generate_id()
            with open(wandb_json_path, "w") as f:
                json.dump({"run_id": run_id}, f)
        wandb.init(
            project="visdiff-finetuning",
            tags=[],
            mode="online",  # Change to "offline" if needed
            id=run_id,
            resume="allow",
            config={"learning_rate": args.learning_rate, "epochs": args.epochs, "batch_size": args.batch_size},
        )
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        wandb.init(mode="offline")  # Fallback to offline mode if there is an initialization failure

    # Setup your model, datasets, etc. here

    # Example:
    DEVICE = "cuda:0"
    USE_LORA = args.training_option == "lora"
    USE_QLORA = args.training_option == "qlora"
    print(f"Lora: {USE_LORA}, QLora: {USE_QLORA}")

    # Assuming this would be the last line of your main setup and training logic
    print("Setup complete, starting training...")

    processor = AutoProcessor.from_pretrained(args.model_name, do_image_splitting=False)

    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
    if args.add_lora_where == "projection":
        target_modules = ".*(modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj).*$"
    elif args.add_lora_where == "projection,resampler":
        target_modules = (
            ".*(modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
        )
    elif args.add_lora_where == "text_model,projection,resampler":
        target_modules = ".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
    else:
        raise ValueError(f"Unknown target modules: {args.add_lora_where}")
    print(f"Target modules: {target_modules}")

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.2,
            target_modules=target_modules,
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian",
        )
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args.model_name,
            device_map=device_map,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config if USE_QLORA else None,
        )
        model.add_adapter(lora_config)
        model.enable_adapters()

    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            _attn_implementation="flash_attention_2",  # Only available on A100 or H100
        ).to(DEVICE)
    print_trainable_parameters(model)

    with open(os.path.join(output_dir, "model.json"), "w") as f:
        f.write(str(model))

    from datasets import load_dataset

    dataset_name = args.dataset_name
    train_dataset = load_dataset(
        "csv", data_files=f"/network/projects/mair_project_vim/annotations/generative/train/train_{dataset_name}.csv"
    )["train"]
    eval_dataset = load_dataset(
        "csv",
        data_files=f"/network/projects/mair_project_vim/annotations/generative/validation/validation_{dataset_name}.csv",
    )["train"]

    """Let's look at an example. Each sample has two images, a question, and an answer."""

    print(train_dataset)
    print(train_dataset[10]["image_path_1"])

    """# Training loop

    We first define the data collator which takes list of samples and return input tensors fed to the model. There are 4 tensors types we are interested:
    - `input_ids`: these are the input indices fed to the language model
    - `attention_mask`: the attention mask for the `input_ids` in the language model
    - `pixel_values`: the (pre-processed) pixel values that encode the image(s). Idefics2 treats images in their native resolution (up to 980) and their native aspect ratio
    - `pixel_attention_mask`: when multiple image(s) are packed into the same sample (or in the batch), attention masks for the images are necessary because of these images can have different sizes and aspect ratio. This masking ensures that the vision encoder properly forwards the images.

    """

    class MyDataCollator:
        def __init__(self, processor):
            self.processor = processor
            self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
                processor.tokenizer.additional_special_tokens.index("<image>")
            ]

        def __call__(self, examples):
            texts = []
            images = []
            for example in examples:
                question = example["question"]
                answer = example["answer"]
                if example["type"] == "multi-image":
                    user_content = [
                        {"type": "image"},
                        {"type": "image"},
                    ]
                    ex_images = [example["image_path_1"], example["image_path_2"]]
                else:
                    user_content = [{"type": "image"}]
                    ex_images = [example["image_path_1"]]

                if question:
                    user_content.append({"type": "text", "text": question})
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
                text = processor.apply_chat_template(messages, add_generation_prompt=False)
                if "base" in args.model_name:  # hack to remove the end of utterance token
                    text = text.replace("<end_of_utterance>", "")
                texts.append(text.strip())
                ex_images = [Image.open(img_path).convert("RGB") for img_path in ex_images]
                images.append(ex_images)

            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
            batch["labels"] = labels

            return batch

    data_collator = MyDataCollator(processor)

    """We will use HuggingFace Trainer."""

    num_devices = torch.cuda.device_count() * args.num_nodes
    gradient_accumulation_steps = max(1, args.batch_size // (args.batch_size_per_device * num_devices))

    model_name_str = args.model_name.replace("/", "-").lower()
    model_id = f"{model_name_str}-{dataset_name}-{args.training_option}-lr{args.learning_rate}"
    output_dir = os.path.join(args.output_dir, model_id)
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_device,
        # per_device_eval_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=10,
        log_level="info",
        output_dir=output_dir,
        overwrite_output_dir=args.resume_from_checkpoint is None,
        save_strategy="steps",
        save_steps=args.save_steps,
        # eval_steps=200,
        save_total_limit=50,
        # evaluation_strategy="steps",
        fp16=True,
        # resume_from_checkpoint=True,
        # push_to_hub_model_id=model_id,
        remove_unused_columns=False,
        report_to="all",
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint is not None:
        from transformers.trainer_utils import get_last_checkpoint

        print(f"Searching for last checkpoint in {output_dir}")
        resume_from_checkpoint = get_last_checkpoint(output_dir)
        if resume_from_checkpoint is not None:
            print(f"Resuming training from {resume_from_checkpoint}")
            model = Idefics2ForConditionalGeneration.from_pretrained(
                resume_from_checkpoint,
                device_map=device_map,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config if USE_QLORA else None,
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,  # You can also evaluate (loss) on the eval set, note that it will incur some additional GPU memory
    )

    """# Training and pushing to the hub

    We have all the core building blocks now, so we fine-tune the model!

    The training can take a few minutes depending on the hardware you use.
    """

    trainer.train(resume_from_checkpoint=resume_from_checkpoint, without_checkpoint_model=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is used for fine-tuning the IDEFICS2 model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceM4/idefics2-8b",
        help="Specify the name of the model to be fine-tuned.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Specify the path to the checkpoint from which the training should be resumed.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="itm_ours",
        # choices=["ours", "nearest","mixed"],
        help="Specify the name of the dataset to be used for fine-tuning the model.",
    )
    parser.add_argument(
        "--training_option",
        type=str,
        default="lora",
        choices=["qlora", "lora", "full"],
        help="Choose the training option: qlora for QLora (lowest precision training), lora for Standard Lora (medium precision training), full for Full fine-tuning (highest precision training)",
    )
    parser.add_argument(
        "--add_lora_where",
        type=str,
        default="text_model,projection,resampler",
        choices=["projection", "projection,resampler", "text_model,projection,resampler"],
        help="Choose the target modules for Lora/QLora adapter.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Specify the number of epochs for training the model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Specify the batch size for training the model.")
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=8,
        help="Specify the batch size per device for training the model.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Specify the learning rate for training the model."
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="linear", help="Specify the learning rate scheduler type."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=300, help="Specify the number of warmup steps for training the model."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Specify the weight decay for training the model."
    )
    parser.add_argument("--lora_r", type=int, default=8, help="Specify the r value for Lora.")
    parser.add_argument("--lora_alpha", type=int, default=8, help="Specify the alpha value for Lora.")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mila/r/rabiul.awal/scratch/synth-diffuse/checkpoints",
        help="Specify the directory where the model checkpoints will be saved.",
    )
    parser.add_argument("--save_steps", type=int, default=300, help="Specify the number of steps to save the model.")
    args = parser.parse_args()
    main(args)
