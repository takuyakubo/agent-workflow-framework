"""
Tool registry module for agent workflow framework.

This module provides a registry for managing and accessing tools.
"""

from typing import Dict, List, Optional

from agent_workflow_framework.tools.base import Tool


class ToolRegistry:
    """
    Registry for managing tools available to agents.
    
    This class provides methods for registering, accessing, and managing tools
    that can be used by agent nodes.
    """
    
    def __init__(self):
        """
        Initialize a new tool registry.
        """
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: The tool to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"A tool with name '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: The name of the tool to get
            
        Returns:
            The tool, or None if no tool with that name is registered
        """
        return self._tools.get(name)
    
    def list(self) -> List[Tool]:
        """
        Get a list of all registered tools.
        
        Returns:
            A list of all registered tools
        """
        return list(self._tools.values())
    
    def to_dict_list(self) -> List[Dict]:
        """
        Convert all registered tools to a list of dictionaries.
        
        This format is suitable for use with LLMs that support tool calling.
        
        Returns:
            A list of dictionaries representing the tools
        """
        return [tool.to_dict() for tool in self._tools.values()]
    
    def __contains__(self, name: str) -> bool:
        """
        Check if a tool with the given name is registered.
        
        Args:
            name: The name of the tool to check for
            
        Returns:
            True if a tool with the given name is registered, False otherwise
        """
        return name in self._tools
