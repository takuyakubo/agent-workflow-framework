"""
Base LLM interface for agent workflow framework.

This module defines the interface that all language model implementations must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLM(ABC):
    """
    Base class for language model implementations.
    
    This abstract class defines the interface that all language model 
    implementations must implement to be used with the agent workflow framework.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the name of the underlying model.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Returns the name of the model provider.
        """
        pass
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the language model with a text prompt.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments for the model
            
        Returns:
            The model's response as a string
        """
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Any:
        """
        Stream the language model's response for a text prompt.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments for the model
            
        Returns:
            A generator or iterator for the streaming response
        """
        pass

    @abstractmethod
    def invoke_with_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Invoke the language model with chat messages.
        
        Args:
            messages: List of message objects with role and content
            **kwargs: Additional arguments for the model
            
        Returns:
            The model's response as a string
        """
        pass
    
    def invoke_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Invoke the language model with tools.
        
        Args:
            prompt: The prompt to send to the model
            tools: A list of tool specifications
            **kwargs: Additional arguments for the model
            
        Returns:
            A dictionary containing the response and any tool calls
        """
        raise NotImplementedError("Tool calling not implemented for this model")
    
    def invoke_with_structured_output(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Invoke the language model with a structured output schema.
        
        Args:
            prompt: The prompt to send to the model
            output_schema: A JSON schema for the expected output
            **kwargs: Additional arguments for the model
            
        Returns:
            A dictionary conforming to the output schema
        """
        raise NotImplementedError("Structured output not implemented for this model")
    
    def get_image_object(self, image_path: str) -> Dict[str, Any]:
        """
        Convert an image path to a format that can be used with the language model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            A dictionary containing the image data in a format that the model can use
        """
        raise NotImplementedError("Image support not implemented for this model")
