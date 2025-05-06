"""
Core abstractions for language models.

This module provides the base classes and interfaces for working with different
language model providers in a unified way.
"""

from abc import ABC


class UnifiedModel(ABC):

    @property
    def provider_name(self) -> str:
        """
        Returns the name of the model provider.
        """
        pass

    def get_image_object(self, image_path) -> dict:
        pass
