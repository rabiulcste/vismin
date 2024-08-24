import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Union

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from commons.constants import (COCO_ONLY_CATEGORIES,
                               MISTRALAI_LANGUAGE_MODEL_NAMES,
                               SYNTH_DIFFUSE_DATA_DIR,
                               VALID_SPATIAL_DIRECTIONS)
from commons.logger import Logger
from commons.utils import load_json_data, save_to_json

from .helpers import get_coco_tags

logger = Logger.get_logger(__name__)

VLLM_MODELS = [
    "lmsys/vicuna-13b-v1.3",
    "meta-llama/Llama-2-70b-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    "google/gemma-7b",
    "google/gemma-7b-it",
    "teknium/OpenHermes-2.5-Mistral-7B",
]


class BaseLLM(ABC):
    """
    A base class that uses a language model to generate output on a given input text from in-context examples.
    """

    def __init__(
        self,
        language_model_name: str,
        prompt_type: str,
        device: str,
    ):
        """
        Initializes the tokenizer and llm using the specified pretrained model.

        Args:
            language_model_name (str): The name of the pretrained model to use.
        """
        logger.info(
            f"Initializing {self.__class__.__name__} with language_model_name: {language_model_name}. It may take a few minutes to load the model."
        )
        logger.info(f"prompt_type: {prompt_type}, device: {device}")

        self.language_model_name = language_model_name
        self.prompt_data = self._load_in_context_data(prompt_type)
        self.device = device
        self.autocast_dtype = self._get_autocast_dtype()
        self.autocast_dtype_str = self._get_autocast_dtype_str()

    @staticmethod
    def _get_autocast_dtype():
        """Return the best available dtype for autocasting: bfloat16 if available, else float16."""
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _get_autocast_dtype_str(self):
        """Return the best available dtype for autocasting: bfloat16 if available, else float16."""
        if torch.cuda.is_bf16_supported() and "gptq" not in self.language_model_name.lower():
            return "bfloat16"
        return "float16"

    def load_pretrained_model_tokenizer(self):
        logger.info(f"Loading pretrained model and tokenizer: {self.language_model_name}")
        self.vllm_enabled = False

        if self.language_model_name in VLLM_MODELS:
            from vllm import LLM

            self.llm = LLM(
                model=self.language_model_name,
                dtype=self.autocast_dtype_str,
                tensor_parallel_size=torch.cuda.device_count(),
            )
            self.vllm_enabled = True
            self.tokenizer = self.llm.get_tokenizer()

        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "</s>"})

            self.llm = AutoModelForCausalLM.from_pretrained(
                self.language_model_name,
                low_cpu_mem_usage=True,
                torch_dtype=self.autocast_dtype,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quantization_config,
            )
            self.vllm_enabled = False

    def tokenize_single_input(tokenizer, prompt):
        # OpenChat V2
        human_prefix = "User:"
        prefix = "Assistant GPT4:"
        eot_token = "<|end_of_turn|>"
        bos_token = "<s>"

        def _tokenize(text):
            return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text))

        def _tokenize_special(special_name):
            return tokenizer.convert_tokens_to_ids(special_name)

        return (
            [_tokenize_special(bos_token)]
            + _tokenize(human_prefix)
            + _tokenize(prompt)
            + [_tokenize_special(eot_token)]
            + _tokenize(prefix)
        )

    @abstractmethod
    def get_prompt_data_fpath(prompt_type):
        pass

    @abstractmethod
    def _prepare_prompt(self, image_caption: str, num_examples_in_task_prompt: int):
        pass

    def _load_in_context_data(self, prompt_type: str):
        fname = self.get_prompt_data_fpath(prompt_type)
        try:
            with open(fname, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
                logger.info(f"Loaded prompt data successfully from {fname}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find prompt source file {fname}. Please check that the file exists in `promptsource` directory."
            )
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Could not decode JSON from {fname}. Please check the file format.")

        logger.info(f"Prompt metadata: {prompt_data['meta_description']}")
        logger.info(f"Prompt IO: {prompt_data['io_structure']}")
        logger.info(
            f"Prompt example: {prompt_data['samples'][0] if isinstance(prompt_data['samples'], list) else None}"
        )

        return prompt_data

    def generate_output(
        self,
        input_text: str = None,
        prompt: str = None,
        max_length: int = 128,
        num_examples_in_task_prompt: int = 6,
        num_return_sequences: int = 1,
        num_beams: int = 1,
    ):
        """
        Generates descriptive output given an input text and a list of examples.

        Args:
            input_text (str): The input text.
            examples (List[str]): The list of examples.
            max_length (int): The maximum length of the generated output.

        Returns:
            A list of generated outputs.
        """
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string.")

        if prompt is None:
            prompt = self._prepare_prompt(input_text, num_examples_in_task_prompt)
        logger.debug(f"Input full prompt: {prompt}")

        encodings = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)
        prompt_length = input_ids.shape[-1]
        generated_ids = self.llm.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=prompt_length + max_length,
            do_sample=True,
            top_k=40,
            temperature=0.8,
            num_return_sequences=num_return_sequences,
        )

        generated_ids = generated_ids[:, prompt_length:]
        generated_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logger.debug(f"Input Text: {input_text}, Generated Output: {generated_outputs}")

        return generated_outputs

    def generate_output_batch(
        self,
        input_texts: List[str] = None,
        all_prompts: List[str] = None,
        max_length: int = 128,
        num_examples_in_task_prompt: int = 8,
        num_return_sequences: int = 1,
        num_beams: int = None,
        top_k: float = 40,
        temperature: float = 0.8,
        stop: str = "\n",
        **kwargs,
    ):
        """
        Generates descriptive tags for a batch of images given their captions.

        Args:
            image_captions (List[str]): The captions of the images.
            max_length (int): The maximum length of the generated tags.

        Returns:
            A list of lists, where each sub-list contains descriptive tags for the corresponding image.
        """

        if input_texts is None and all_prompts is None:
            raise ValueError("Either input_texts or all_prompts must be provided.")

        if all_prompts is None:
            if not all(isinstance(input_text, str) for input_text in input_texts):
                logger.warning("All image captions must be strings.")
                return []
            all_prompts = [
                self._prepare_prompt(input_text, num_examples_in_task_prompt, **kwargs) for input_text in input_texts
            ]

        if self.vllm_enabled:
            from vllm import SamplingParams

            if num_beams is not None:
                sampling_params = SamplingParams(
                    n=num_return_sequences,
                    max_tokens=max_length,
                    best_of=num_beams,
                    stop=stop,
                    temperature=0.0,
                    use_beam_search=True,
                )
            else:
                sampling_params = SamplingParams(
                    n=num_return_sequences,
                    best_of=num_return_sequences,
                    max_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    stop=stop,
                )
            generated_outputs = self.llm.generate(
                prompts=all_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            generated_outputs = [[out.text for out in output.outputs] for output in generated_outputs]
            generated_outputs = [item for sublist in generated_outputs for item in sublist]

        else:
            encodings = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)

            prompt_length = input_ids.shape[-1]

            generated_ids = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=prompt_length + max_length,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
            )
            generated_ids = generated_ids[:, prompt_length:]
            generated_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        logger.debug(f"Input Text: {input_texts}, Generated Output: {generated_outputs}")

        return generated_outputs


# given an input object, find some words that can be useful to describe it
class ImageNetCaptionerLLM(BaseLLM):
    """
    LLM for generating captions for ImageNet classes.
    """

    NUM_RETURN_SEQUENCES = 20

    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        return os.path.join("promptsource", f"imagenet_{prompt_type}.json")

    def _prepare_prompt(self, image_caption: str, num_examples_in_task_prompt: int):
        examples = self.prompt_data["samples"]

        random.shuffle(examples)
        selected_examples = random.sample(examples, num_examples_in_task_prompt)

        # Create the final prompt instruction with correct numbering
        prompt_instruction = self.prompt_data["task_instruction"]
        for i, example in enumerate(selected_examples, start=1):
            prompt_instruction += f"{i}. {example}\n"
        prompt_instruction += f"{len(selected_examples) + 1}. {{input_key}}: {{input_text}} {{output_key}}:"

        prompt = prompt_instruction.format(
            input_key=self.prompt_data["io_structure"]["input_keys"],
            input_text=image_caption,
            output_key=self.prompt_data["io_structure"]["output_keys"],
        )
        return prompt

    def generate_caption(self, input_text: Union[List[str], str]):
        if isinstance(input_text, list):
            return self.generate_output_batch(input_texts=input_text, num_return_sequences=self.NUM_RETURN_SEQUENCES)
        return self.generate_output(input_text=input_text, num_return_sequences=self.NUM_RETURN_SEQUENCES)

    def generate_descriptive_tags(self, image_caption: str):
        return self.generate_output(input_text=image_caption, num_return_sequences=self.NUM_RETURN_SEQUENCES)

    def refine_web_caption(self, web_caption: Union[List[str], str]):
        if isinstance(web_caption, list):
            return self.generate_output_batch(input_texts=web_caption)
        return self.generate_output(input_text=web_caption, num_return_sequences=1)


class InstructPix2PixLLM(BaseLLM):
    """
    A class that uses a language model to generate an edit instruction and a new caption on an input caption.
    """

    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        return os.path.join("promptsource", f"{prompt_type}.json")

    def _prepare_prompt(self, image_caption: str, num_examples_in_task_prompt: int):
        examples = self.prompt_data["samples"]

        random.shuffle(examples)
        selected_examples = random.sample(examples, num_examples_in_task_prompt)

        # Create the final prompt instruction with correct numbering
        prompt_instruction = self.prompt_data["task_instruction"]
        for i, example in enumerate(selected_examples, start=1):
            prompt_instruction += f"{i}. {example}\n"
        prompt_instruction += f"{len(selected_examples) + 1}. {{input_key}}: {{image_caption}}."

        prompt = prompt_instruction.format(
            input_key=self.prompt_data["io_structure"]["input_keys"],
            image_caption=image_caption,
        )
        return prompt

    # TODO: This is not properly tested yet, so needs to be tested before using
    def generate_edit_instructions_batch(self, image_captions: List[str], max_length=50):
        """
        Generates descriptive tags for a batch of images given their captions.

        Args:
            image_captions (List[str]): The captions of the images.
            max_length (int): The maximum length of the generated tags.

        Returns:
            A list of lists, where each sub-list contains descriptive tags for the corresponding image.
        """
        if not all(isinstance(caption, str) for caption in image_captions):
            raise ValueError("All image captions must be strings.")

        # Create a list to hold all prompts
        all_prompts = [self._prepare_prompt(caption) for caption in image_captions]

        # Tokenize all prompts and prepare tensor for the model
        input_ids = self.tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(
            self.device
        )

        # Generate output for all prompts
        model_output = self.model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=10,
        )

        # Decode the output
        decoded_output = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

        # Create a list to store the final results
        all_edit_instructions = []

        for i, output in enumerate(decoded_output):
            text = output[len(all_prompts[i]) :]
            split_text = text.split(f"{self.num_examples_in_task_prompt+2}. Input:")
            if len(split_text) > 1:
                desired_text = split_text[0]
            else:
                desired_text = text
            all_edit_instructions.append(desired_text.replace("\n", "").strip())

        return all_edit_instructions


class DiffEditLLM(BaseLLM):
    """
    A class that uses a language model to generate an edit instruction and a new caption on an input caption.
    """

    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        return os.path.join("promptsource", f"{prompt_type}.json")

    def _prepare_prompt(self, image_caption: str, num_examples_in_task_prompt: int):
        examples = self.prompt_data["samples"]

        random.shuffle(examples)
        selected_examples = random.sample(examples, num_examples_in_task_prompt)

        # Create the final prompt instruction with correct numbering
        prompt_instruction = self.prompt_data["task_instruction"]
        for i, example in enumerate(selected_examples, start=1):
            prompt_instruction += f"{i}. {example}\n"
        prompt_instruction += f"{len(selected_examples) + 1}. {{input_key}}: {{image_caption}}."

        prompt = prompt_instruction.format(
            input_key=self.prompt_data["io_structure"]["input_keys"],
            image_caption=image_caption,
        )
        return prompt


class EditsGenLLM(BaseLLM):
    """
    A class that uses a language model to generate an edit instruction and a new caption on an input caption.
    """

    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root_dir, "promptsource", f"{prompt_type}.json")

    def get_json_formatted_examples(self, selected_examples: List[str], input_text: str):
        formatted_examples_json = "\n".join(str(example) for example in selected_examples) + "\n"
        test_example = (
            "Now, write multiple, distinct and diverse edits strictly following the task criteria. "
            'The output must be a list in the shown JSON format, for "{input_key}: {input_text}". '
        )
        # add full stop to the end of the image_caption if it doesn't have one
        if not input_text.strip().endswith("."):
            input_text = input_text.strip() + "."

        test_example = test_example.format(
            input_key=self.prompt_data["io_structure"]["input_keys"],
            input_text=input_text,
        )
        return f"{formatted_examples_json}{test_example}"

    def get_no_attribute_found(self):
        return [
            {
                "InputCaption": "A cat is on the floor.",
                "SelectedPhrase": "No attribute change",
                "EditedPhrase": "No attribute change",
                "EditedRegionPhrase": "No attribute change",
                "EditedCaption": "No attribute change",
                "Category": "No attribute change",
            },
            {
                "InputCaption": "A glass of water and a sandwich.",
                "SelectedPhrase": "No attribute change",
                "EditedPhrase": "No attribute change",
                "EditedRegionPhrase": "No attribute change",
                "EditedCaption": "No attribute change",
                "Category": "No attribute change",
            },
            {
                "InputCaption": "A dog is sitting on a table.",
                "SelectedPhrase": "No attribute change",
                "EditedPhrase": "No attribute change",
                "EditedRegionPhrase": "No attribute change",
                "EditedCaption": "No attribute change",
                "Category": "No attribute change",
            },
            {
                "InputCaption": "A vase filled with flowers and leaves.",
                "SelectedPhrase": "No attribute change",
                "EditedPhrase": "No attribute change",
                "EditedRegionPhrase": "No attribute change",
                "EditedCaption": "No attribute change",
                "Category": "No attribute change",
            },
        ]

    def _prepare_prompt(self, input_text: str, num_examples_in_task_prompt: int, selected_category: str):
        samples = self.prompt_data["samples"]
        selected_examples = random.sample(samples[selected_category], num_examples_in_task_prompt)
        if selected_category == "attribute":
            selected_examples += random.sample(self.get_no_attribute_found(), 2)

        random.shuffle(selected_examples)

        task_instruction = self.prompt_data[f"task_instruction_{selected_category}"] + "\n"
        examples_with_test = self.get_json_formatted_examples(selected_examples, input_text)
        final_prompt = f"{task_instruction}{examples_with_test}"

        # TODO: test [INST] and [/INST formatting for Mixtral
        if self.language_model_name in MISTRALAI_LANGUAGE_MODEL_NAMES:  # if using Mixtral
            final_prompt = f"[INST] {final_prompt} [/INST]"
        return final_prompt


# Note: this is a little different from the other LLMs because it uses a different prompt structure
class GroundedLLM(BaseLLM):
    """
    A class that uses a language model to generate an edit instruction and a new caption on an input caption.
    """

    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root_dir, "promptsource", f"{prompt_type}.json")

    @staticmethod
    def parse_and_format(input_str):
        pattern = r"Input: (.*?)\.\nBounding boxes: (\[.*?\])\nBackground prompt: (.*?)\nNegative prompt:(.*?)\nCategory: (.*?)$"
        match = re.match(pattern, input_str, re.DOTALL)

        if match:
            input_description = match.group(1).strip()
            bounding_boxes = match.group(2).strip()
            background_prompt = match.group(3).strip()
            negative_prompt = match.group(4).strip()
            category = match.group(5).strip()

            comma_separated = f"Input: {input_description}, Bounding boxes: {bounding_boxes}, Background prompt: {background_prompt}, Negative prompt: {negative_prompt}, Category: {category}"
            return comma_separated

        return None

    def load_valid_coco_nouns(
        self,
    ):
        _, _, categories_name_id, _ = get_coco_tags("train")
        self.coco_nouns = list(categories_name_id.keys())
        print(self.coco_nouns)

    def _prepare_prompt(self, image_caption: str, num_examples_in_task_prompt: int, dataset: str):
        sampled_examples_dict = {}
        category_data = self.prompt_data["samples"][dataset]
        for category, examples_dict in category_data.items():
            if category not in VALID_SPATIAL_DIRECTIONS:
                continue
            safe_sample_size = min(8, len(examples_dict))
            sampled_items_dict = dict(random.sample(list(examples_dict.items()), safe_sample_size))
            sampled_examples_dict.update(sampled_items_dict)

        examples = []
        for input_text, output_texts in sampled_examples_dict.items():
            input_part = f"{self.prompt_data['io_structure']['input_keys']}: {input_text + '.' if not input_text.endswith('.') else input_text}"
            output_part = f"Bounding boxes: {random.choice(output_texts)}"
            curr_inp_str = f"{input_part}\n{output_part}"
            formatted_example_curr = self.parse_and_format(curr_inp_str)
            if formatted_example_curr:
                examples.append(formatted_example_curr)

        selected_examples = random.sample(examples, num_examples_in_task_prompt)
        random.shuffle(selected_examples)
        logger.debug(f"Selected examples: {selected_examples}")
        # Create the final prompt instruction with correct numbering
        task_instruction = self.prompt_data[f"task_instruction_{dataset}"] + "\n"
        logger.debug(f"Choose dataset: {dataset}")
        formatted_examples = ""
        for i, example in enumerate(selected_examples, start=1):
            formatted_examples += f"{i}. {example}\n"

        prompt = f"{task_instruction}{formatted_examples}"
        logger.debug(f"Final prompt: {prompt}")
        if self.language_model_name in MISTRALAI_LANGUAGE_MODEL_NAMES:
            prompt = f"[INST] {prompt} [/INST]"
        return prompt


class SkillAnnotator(BaseLLM):
    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        return os.path.join("promptsource", f"{prompt_type}.json")

    def _prepare_prompt(self, image_caption: str, num_examples_in_task_prompt: int):
        examples = self.prompt_data["samples"]

        examples = [
            f"{self.prompt_data['io_structure']['input_keys']}: {key} {self.prompt_data['io_structure']['output_keys']}: {random.choice(value)}"
            for key, value in examples.items()
        ]
        selected_examples = random.sample(examples, num_examples_in_task_prompt)

        # Create the final prompt instruction with correct numbering
        prompt_instruction = self.prompt_data["task_instruction"]
        for i, example in enumerate(selected_examples, start=1):
            prompt_instruction += f"{i}. {example}\n"
        prompt_instruction += f"{len(selected_examples) + 1}. {{input_key}}: {{image_caption}}"

        prompt = prompt_instruction.format(
            input_key=self.prompt_data["io_structure"]["input_keys"],
            image_caption=image_caption,
        )
        return prompt


def process_tifa_data(fpath: str) -> list:
    """Process and format the data from the TIFA JSON file."""
    data = load_json_data(fpath)
    processed_data = defaultdict(list)
    for item in data:
        processed_data[item["id"]].append(
            {
                "caption": item["caption"],
                "question": item["question"],
                "answer": item["answer"],
                "choices": item["choices"],
            }
        )

    formatted_data_list = []
    for items in processed_data.values():
        formatted_data = {
            "Input": items[0]["caption"],
            "Output": [
                f"Question: {item['question']} Choices: {item['choices']} Answer: {item['answer']}" for item in items
            ],
        }
        formatted_data_list.append(formatted_data)
    return formatted_data_list


def save_tifa_prompt_demo_data(output_filepath: str):
    tifa_filepath = "/home/mila/r/rabiul.awal/synth-diffuse/tifa/tifa_v1.0/tifa_v1.0_question_answers.json"
    samples = process_tifa_data(tifa_filepath)
    logger.debug(json.dumps(samples[0], indent=4))

    prompt_data = {
        "meta_description": "Generate questions and answers from image captions.",
        "task_instruction_full_caption": (
            "As an AI language model, your task is to:\n"
            "- Generate relevant questions and their corresponding answers based on the provided image caption.\n"
            "- Focus primarily on the initial context or the first sentence of the caption to guide your question formulation.\n"
            "- Ensure all questions are directly tied to the information given in the caption, and that the answers can be clearly derived from it.\n"
            "- Try to create multiple-choice questions. Each multiple-choice question should have one correct answer and at least three incorrect options. The incorrect options should be plausible but clearly incorrect based on the information in the caption.\n"
            "- Additionally, focus on verifying both the presence and absence of the key context as stated in the caption i.e. object or attribute by formulating binary questions.\n"
            "A few examples of question-answer pairs are provided below for your reference:\n"
        ),
        "task_instruction_region_caption": (
            "As an AI language model, your task is to:\n"
            "- Generate relevant questions and their corresponding answers based on the provided image caption.\n"
            "- Create separate binary questions to confirm both the presence and absence of objects or attributes. "
            "One question should be designed to confirm the presence of the object or attribute (with the expected answer being 'yes'), "
            "and another question should be designed to confirm the absence of the object or attribute (with the expected answer being 'no').\n"
            "- Also, formulate one multiple-choice question related to the objects or attributes.\n\n"
            "A few examples of question-answer pairs are provided below for your reference:\n"
        ),
        "task_instruction_region_obj_count_caption": (
            "As an AI language model, your task is to:\n"
            "- Generate questions that primarily focus on counting the objects depicted in the provided image caption.\n"
            "- While the main focus is on counting, you can also include questions in a mix of multiple-choice and binary (yes/no) format to verify the presence and attributes of objects.\n"
            "- Try to create as many counting-related questions as possible.\n"
            "Below are a few examples of question-answer pairs for your reference:\n"
        ),
        "task_instruction_region_neg_only_caption": (
            "As an AI language model, your task is to:\n"
            "- Generate relevant questions and their corresponding answers based on the provided image caption.\n"
            "- Create binary questions to confirm absence of objects or attributes. "
            "A few examples of question-answer pairs are provided below for your reference:\n"
        ),
        "io_structure": {"input_keys": "Input", "output_keys": "Output"},
        "samples": samples,
    }

    save_to_json(prompt_data, output_filepath)
