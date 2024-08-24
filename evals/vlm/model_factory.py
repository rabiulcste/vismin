import logging
from functools import partial

from commons.constants import GEMINI_API_KEY, OPENAI_API_KEY
from commons.logger import Logger

from .blip import BlipModel
from .blip2 import Blip2Model
from .clip import ClipModel
from .cogvlm import CogVLMModel
from .flava import FlavaModel
from .idefics2 import Idefics2Model
from .internvl import InternVLModel
from .llava import LlaVaModel
from .mantis import MantisModel
from .siglip import SiglipModel
from .vision_api import GeminiModel, GPT4VisionModel

logger = Logger.get_logger(__name__)


class ModelFactory:
    """
    Factory class to manage and instantiate supported models.
    """

    MLLM_MODELS = {
        "gemini": partial(GeminiModel, api_key=GEMINI_API_KEY),
        "gpt-4-vision-preview": partial(GPT4VisionModel, api_key=OPENAI_API_KEY),
        "HuggingFaceM4/idefics2-8b": partial(Idefics2Model, model_name_or_path="HuggingFaceM4/idefics2-8b"),
        "TIGER-Lab/Mantis-8B-siglip-llama3": partial(
            MantisModel, model_name_or_path="TIGER-Lab/Mantis-8B-siglip-llama3"
        ),
        "llava-hf/llava-v1.6-mistral-7b-hf": partial(
            LlaVaModel, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf"
        ),
        "THUDM/cogvlm-chat-hf": partial(CogVLMModel, model_name="THUDM/cogvlm-chat-hf"),
        "OpenGVLab/InternVL-Chat-V1-5": partial(InternVLModel, model_name="OpenGVLab/InternVL-Chat-V1-5"),
        "Salesforce/blip2-flan-t5-xxl": partial(Blip2Model, model_name_or_path="Salesforce/blip2-flan-t5-xxl"),
    }

    CLIP_MODELS = {
        "ViT-B/32": partial(ClipModel, model_name="ViT-B/32"),
        "ViT-B/16": partial(ClipModel, model_name="ViT-B/16"),
        "ViT-L-14": partial(ClipModel, model_name="ViT-L-14"),
        "facebook/flava-full": partial(FlavaModel, model_name="facebook/flava-full"),
        "blip-coco-base": partial(BlipModel, model_name_or_path="blip-coco-base"),
        "google/siglip-base-patch16-256": partial(SiglipModel, model_name="google/siglip-base-patch16-256"),
        "google/siglip-large-patch16-384": partial(SiglipModel, model_name="google/siglip-large-patch16-384"),
    }

    SUPPORTED_MODELS = {**MLLM_MODELS, **CLIP_MODELS}

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = self.get_model(model_name, **kwargs)

    def is_model_supported(self, model_name: str) -> bool:
        """
        Check if the given model name is supported.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model is supported, False otherwise.
        """
        is_supported = any(model in model_name for model in self.SUPPORTED_MODELS)
        logger.debug(f"Model '{model_name}' supported: {is_supported}")
        return is_supported

    def get_model(self, model_name: str, **kwargs):
        """
        Get an instance of the specified model.

        Args:
            model_name (str): The name of the model to instantiate.

        Returns:
            An instance of the specified model.

        Raises:
            ValueError: If the model name is not supported.
        """
        pretrained = kwargs.pop("pretrained", None)
        if not self.is_model_supported(model_name):
            logger.error(f"Unsupported model name: {model_name}")
            raise ValueError(f"Unsupported model name: {model_name}")
        if pretrained:
            model = self.SUPPORTED_MODELS[model_name](pretrained=pretrained, **kwargs)
        else:
            model = self.SUPPORTED_MODELS[model_name](**kwargs)
        logger.info(f"Model '{model_name}' instantiated")
        return model

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_batch(self, *args, **kwargs):
        return self.model.predict_batch(*args, **kwargs)
