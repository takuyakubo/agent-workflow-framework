"""
Base state module for the agent workflow framework.

This module provides the base state class that all node states should inherit from.
"""

import logging
from typing import Dict, Optional, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeState(BaseModel):
    """
    Base state class for all node states in the workflow.

    This class provides error handling and context management capabilities.
    All workflow states should inherit from this class.
    """

    error: str = Field(default="", description="Error message (if any)")
    context: Dict[str, str] = Field(default_factory=dict, description="Shared context between nodes")

    def emit_error(self, error_str: str) -> "NodeState":
        """
        Create a new state instance with an error message.

        Args:
            error_str: The error message to set

        Returns:
            A new state instance with the error message
        """
        logger.error(f"Node error: {error_str}")
        return self.model_copy(update={"error": error_str})
    
    def add_to_context(self, key: str, value: str) -> None:
        """
        Add a key-value pair to the context.

        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
    
    def get_from_context(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a value from the context by key.

        Args:
            key: Context key
            default: Default value if key is not found

        Returns:
            The value from the context, or the default if not found
        """
        return self.context.get(key, default)


T = TypeVar("T", bound=NodeState)
