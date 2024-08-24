import os
import re
import string
from statistics import mean
from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from commons.logger import Logger
from evals.vlm.model_factory import ModelFactory

logger = Logger.get_logger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class SBERTModel:
    def __init__(self, ckpt="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            print("Using SBERT on GPU")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_sentences(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.model.device))

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.detach().cpu()

    def multiple_choice(self, answer, choices):
        answer_embedding = self.embed_sentences([answer])
        choices_embedding = self.embed_sentences(choices)
        top_choice_index = torch.argmax(torch.matmul(choices_embedding, answer_embedding.T)).item()
        return choices[top_choice_index]


class VQAModelForTifa:
    def __init__(self, model_name_or_path: str, load_in_Nbit: Union[int, None] = None):
        logger.info(
            f"Initializing {__class__.__name__} with model: {model_name_or_path} and load_in_Nbit: {load_in_Nbit}"
        )
        self.model = ModelFactory(model_name=model_name_or_path, load_in_Nbit=load_in_Nbit)
        self.sbert_model = SBERTModel()

    def get_tifa_score(self, question_answer_pairs, image, enable_logging=False):
        prompts = [
            (
                f"Answer the following {'multiple-choice ' if pair['choices'] else ''}question.\n"
                f"Question: {pair['question']}\n"
                "Please select the correct answer from the choices provided below.\n"
                f"{'Choices: ' + ', '.join(pair['choices']) + '.'}\n"
                "Answer:"
            )
            for pair in question_answer_pairs
        ]
        images = [image.convert("RGB")] * len(prompts)
        batch_answers = self.model.predict_batch(question_answer_pairs, images, enable_logging)
        vqa_answers = self.process_mc_answer_batch(question_answer_pairs, batch_answers)

        if enable_logging:
            vqa_output_list = [re.sub("\n+", " ", f"{prompt} {answer}") for prompt, answer in zip(prompts, vqa_answers)]
            formatted_vqa_output_list = "\n".join(f"- {item}" for item in vqa_output_list[:5])
            logger.info(f"Sampled vqa answers (out of {len(vqa_output_list)}) :\n{formatted_vqa_output_list}")

        result_dict = self.compute_score(question_answer_pairs, vqa_answers)
        return result_dict

    def process_mc_answer_batch(self, question_answer_pairs, batch_answers):
        vqa_answers_cleaned = []
        for qap, vqa_answer in zip(question_answer_pairs, batch_answers):
            vqa_answer_norm = self.extract_answer_string(vqa_answer)
            choices = qap["choices"]
            if vqa_answer_norm not in choices:
                vqa_answer_sbert = self.sbert_model.multiple_choice(vqa_answer_norm, choices)
                print(f"VQA answer: '{vqa_answer_norm}' not in choices: {choices}, using SBERT: {vqa_answer_sbert}")
                vqa_answer_norm = vqa_answer_sbert
            vqa_answers_cleaned.append(vqa_answer_norm)
        return vqa_answers_cleaned

    @staticmethod
    def extract_answer_string(text):
        pattern = r"(.*?)(Answer the|Please answer the following|Answer the multiple-choice question|Answer:|USER:|Explanation:|Question:)"
        match = re.search(pattern, text, re.DOTALL)
        answer = None
        if match:
            # If a match is found, return the captured group (the text before any of the phrases), stripped and in lower case
            answer = match.group(1)
        else:
            # Handle the case where the text contains exactly two lines
            lines = re.split("\n|,", text)
            if len(lines) >= 2:
                answer = lines[0]

        if answer is None:
            answer = text
        answer = answer.lower().strip(string.punctuation + string.whitespace)
        return answer

    def compute_score(self, question_answer_pairs, vqa_answers):
        question_logs = {}
        for free_form_answer, question_answer_pair in zip(vqa_answers, question_answer_pairs):
            if question_answer_pair["question"] not in question_logs:
                question_logs[question_answer_pair["question"]] = question_answer_pair

            mc_answer = free_form_answer
            question_logs[question_answer_pair["question"]]["multiple_choice_vqa"] = mc_answer

            # compute multiple choice score
            score = int(mc_answer == question_answer_pair["answer"])
            question_logs[question_answer_pair["question"]]["scores"] = score

        tifa_score = mean(
            [
                int(vqa_answer == question_answer_pair["answer"])
                for vqa_answer, question_answer_pair in zip(vqa_answers, question_answer_pairs)
            ]
        )
        result_dict = {"tifa_score": tifa_score, "question_details": question_logs}

        return result_dict
