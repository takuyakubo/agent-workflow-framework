"""
Anthropic Claude model implementation.
"""

from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic

from ..models import UnifiedModel
from ..utils import image_path_to_image_data

provider_name = "anthropic"
model_prefix = "claude-"


def get_available_models():
    client = Anthropic()
    try:
        data = client.models.list().data
        return [d.id for d in data if d.type == "model"]
    except:
        return []


provided_models = get_available_models()


class AnthropicModel(ChatAnthropic, UnifiedModel):
    """
    Implementation of the unified model interface for Anthropic Claude models.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Anthropic model.

        Args:
            model_name: Claude model name
            **kwargs: Additional arguments for the model
        """
        super(ChatAnthropic, self).__init__(model=model_name, **kwargs)

    @property
    def provider_name(self) -> str:
        """
        Returns the name of the model provider.
        """
        return provider_name

    @staticmethod
    def get_image_object(image_path) -> dict:
        mime_type, image_data = image_path_to_image_data(image_path)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": image_data},
        }


model_class = AnthropicModel


def get_provider_info():
    """
    Returns provider registration information.

    Returns:
        dict: Provider information with keys:
            - name: Provider name
            - model_class: The model implementation class
            - model_prefix: Prefix for model names (if any)
            - custom_models: List of custom models (if any)
    """
    return {
        "name": provider_name,
        "model_class": model_class,
        "model_prefix": model_prefix,
        "custom_models": provided_models,
    }
