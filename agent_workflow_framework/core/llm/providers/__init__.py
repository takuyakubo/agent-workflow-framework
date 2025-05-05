from enum import Enum
from typing import Dict, Type

from ..models import UnifiedModel
from . import anthropic, google, lmstudio, openai

# プロバイダ情報を集約; get_providerが呼ばれた場合、この順序が優先されるので注意すること
# e.g. gemma-3-12b-itが googleとlmstudioにある場合、先にあるlmstudio優先

_providers = [
    anthropic.get_provider_info(),
    lmstudio.get_provider_info(),
    google.get_provider_info(),
    openai.get_provider_info(),
]

"""
Provider-specific implementations of the unified model interface.
"""

# ProviderType Enumを作成
ProviderType = Enum("ProviderType", {p["name"].upper(): p["name"] for p in _providers})

# モデルレジストリを構築
model_registry: Dict[str, Type[UnifiedModel]] = {
    p["name"]: p["model_class"] for p in _providers
}

# ToDo: 毎回起動のたびにmodel listのAPIが呼ばれるのは良くないのでキャッシュ機構を検討すること。
allowed_models = {p["name"]: p["custom_models"] for p in _providers}


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
    for p in _providers:
        provider = p["name"]
        # 接頭辞でチェック
        prefix = p["model_prefix"]
        if prefix is not None and model_name.startswith(prefix):
            return provider
        # カスタムモデルでチェック
        custom_models = p["custom_models"]
        if model_name in custom_models:
            return provider

    # 見つからない場合はエラー
    raise ValueError(f"Cannot determine provider for model: {model_name}")
