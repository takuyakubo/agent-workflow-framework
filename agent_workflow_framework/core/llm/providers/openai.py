"""
OpenAI model implementation.
"""

from langchain_openai import ChatOpenAI
from openai import OpenAI

from ..models import UnifiedModel
from ..utils import image_path_to_image_data

provider_name = "openai"
model_prefix = "gpt-"


def get_available_models():
    client = OpenAI()
    try:
        data = client.models.list().data
        return [d.id for d in data if d.object == "model"]
    except:
        return []


provided_models = get_available_models()


class OpenAIModel(ChatOpenAI, UnifiedModel):
    """
    Implementation of the unified model interface for OpenAI models.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the OpenAI model.

        Args:
            model_name: OpenAI model name
            **kwargs: Additional arguments for the model
        """
        super(ChatOpenAI, self).__init__(model=model_name, **kwargs)

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
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        }


model_class = OpenAIModel


# プロバイダ登録情報を返す関数
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
