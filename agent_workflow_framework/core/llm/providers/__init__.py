from enum import Enum

from .anthropic import AnthropicModel
from .anthropic import provider_name as apn
from .google import GoogleModel
from .google import provider_name as gpn
from .openai import OpenAIModel
from .openai import provider_name as opn
from .lmstudio import LMStudioModel
from .lmstudio import provider_name as lms
from .lmstudio import provided_models as lms_models

"""
Provider-specific implementations of the unified model interface.
"""


class ProviderType(Enum):
    """
    Enumeration of supported LLM providers.
    """

    ANTHROPIC = apn
    OPENAI = opn
    GOOGLE = gpn
    LMSTUDIO = lms


model_registory = {
    ProviderType.ANTHROPIC.value: AnthropicModel,
    ProviderType.OPENAI.value: OpenAIModel,
    ProviderType.GOOGLE.value: GoogleModel,
    ProviderType.LMSTUDIO.value: LMStudioModel
}


def get_provider(model_name: str) -> str:
    """
    Determine the provider from the model name.

    Args:
        model_name: Name of the model

    Returns:
        Provider name

    Raises:
        ValueError: If the provider cannot be determined
    """
    if model_name.startswith("claude-"):
        return ProviderType.ANTHROPIC.value
    elif model_name.startswith("gemini-"):
        return ProviderType.GOOGLE.value
    elif model_name.startswith("gpt-"):
        return ProviderType.OPENAI.value
    elif model_name in lms_models:
        return ProviderType.LMSTUDIO.value
    else:
        raise ValueError(f"Cannot determine provider for model: {model_name}")
