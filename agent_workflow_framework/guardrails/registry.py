"""
Guardrail registry module for agent workflow framework.

This module provides a registry for managing and accessing guardrails.
"""

from typing import Dict, List, Optional

from agent_workflow_framework.guardrails.base import Guardrail


class GuardrailRegistry:
    """
    Registry for managing guardrails.
    
    This class provides methods for registering, accessing, and managing guardrails
    that can be applied to agent outputs.
    """
    
    def __init__(self):
        """
        Initialize a new guardrail registry.
        """
        self._guardrails: Dict[str, Guardrail] = {}
    
    def register(self, guardrail: Guardrail) -> None:
        """
        Register a guardrail with the registry.
        
        Args:
            guardrail: The guardrail to register
            
        Raises:
            ValueError: If a guardrail with the same name is already registered
        """
        if guardrail.name in self._guardrails:
            raise ValueError(f"A guardrail with name '{guardrail.name}' is already registered")
        self._guardrails[guardrail.name] = guardrail
    
    def get(self, name: str) -> Optional[Guardrail]:
        """
        Get a guardrail by name.
        
        Args:
            name: The name of the guardrail to get
            
        Returns:
            The guardrail, or None if no guardrail with that name is registered
        """
        return self._guardrails.get(name)
    
    def list(self) -> List[Guardrail]:
        """
        Get a list of all registered guardrails.
        
        Returns:
            A list of all registered guardrails
        """
        return list(self._guardrails.values())
    
    def apply_all(self, output: str) -> str:
        """
        Apply all registered guardrails to an output.
        
        Args:
            output: The output to validate and fix
            
        Returns:
            The fixed output
        """
        current_output = output
        for guardrail in self._guardrails.values():
            validation_result = guardrail.validate(current_output)
            if not validation_result["valid"]:
                current_output = guardrail.fix(current_output, validation_result)
        
        return current_output
    
    def __contains__(self, name: str) -> bool:
        """
        Check if a guardrail with the given name is registered.
        
        Args:
            name: The name of the guardrail to check for
            
        Returns:
            True if a guardrail with the given name is registered, False otherwise
        """
        return name in self._guardrails
