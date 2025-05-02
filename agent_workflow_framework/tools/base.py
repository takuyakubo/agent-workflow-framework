"""
Base tool module for agent workflow framework.

This module provides the base Tool class that all tools should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class Tool(ABC):
    """
    Base class for all tools that can be used by agent nodes.
    
    Tools provide functionality that can be used by the agent to perform actions
    beyond just generating text.
    """
    
    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a new tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function to call when the tool is invoked
        """
        self.name = name
        self.description = description
        self.function = function
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary format that can be used with LLMs.
        
        Returns:
            A dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
        }
    
    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """
        Call the tool with the provided arguments.
        
        Args:
            **kwargs: Arguments for the tool
            
        Returns:
            The result of the tool execution
        """
        return self.function(**kwargs)


class JSONSchemaTool(Tool):
    """
    Tool that uses JSON Schema to define the expected input.
    """
    
    def __init__(self, name: str, description: str, function: Callable, parameters: Dict[str, Any]):
        """
        Initialize a new JSONSchemaTool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function to call when the tool is invoked
            parameters: JSON Schema for the expected parameters
        """
        super().__init__(name, description, function)
        self.parameters = parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary format that can be used with LLMs.
        
        Returns:
            A dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
    
    def __call__(self, **kwargs) -> Any:
        """
        Call the tool with the provided arguments.
        
        Args:
            **kwargs: Arguments for the tool
            
        Returns:
            The result of the tool execution
        """
        return self.function(**kwargs)
