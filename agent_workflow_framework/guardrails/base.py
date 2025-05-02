"""
Base guardrail module for agent workflow framework.

This module provides the base Guardrail class that all guardrails should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union


class Guardrail(ABC):
    """
    Base class for all guardrails that can be applied to agent outputs.
    
    Guardrails provide safety and quality controls for agent outputs.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a new guardrail.
        
        Args:
            name: The name of the guardrail
            description: A description of what the guardrail does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def validate(self, output: str) -> Dict[str, Any]:
        """
        Validate the output against the guardrail.
        
        Args:
            output: The output to validate
            
        Returns:
            A dictionary containing the validation result
        """
        pass
    
    @abstractmethod
    def fix(self, output: str, validation_result: Dict[str, Any]) -> str:
        """
        Fix the output based on the validation result.
        
        Args:
            output: The original output
            validation_result: The result of the validation
            
        Returns:
            The fixed output
        """
        pass


class RegexGuardrail(Guardrail):
    """
    Guardrail that validates outputs using regular expressions.
    """
    
    def __init__(self, name: str, description: str, patterns: List[Dict[str, Any]]):
        """
        Initialize a new regex guardrail.
        
        Args:
            name: The name of the guardrail
            description: A description of what the guardrail does
            patterns: A list of patterns to check for, each with:
                - pattern: The regex pattern
                - action: "allow" or "block"
                - message: Message to include in validation result
        """
        super().__init__(name, description)
        self.patterns = patterns
    
    def validate(self, output: str) -> Dict[str, Any]:
        """
        Validate the output using regex patterns.
        
        Args:
            output: The output to validate
            
        Returns:
            A dictionary containing the validation result
        """
        import re
        
        issues = []
        for pattern_info in self.patterns:
            pattern = pattern_info["pattern"]
            action = pattern_info["action"]
            message = pattern_info["message"]
            
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                if action == "block":
                    issues.append({
                        "pattern": pattern,
                        "matches": matches,
                        "message": message
                    })
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def fix(self, output: str, validation_result: Dict[str, Any]) -> str:
        """
        Fix the output by removing or replacing problematic content.
        
        Args:
            output: The original output
            validation_result: The result of the validation
            
        Returns:
            The fixed output
        """
        import re
        
        fixed_output = output
        if not validation_result["valid"]:
            for issue in validation_result["issues"]:
                pattern = issue["pattern"]
                fixed_output = re.sub(pattern, "[REDACTED]", fixed_output, flags=re.IGNORECASE)
        
        return fixed_output


class SchemaGuardrail(Guardrail):
    """
    Guardrail that validates outputs against a JSON schema.
    """
    
    def __init__(self, name: str, description: str, schema: Dict[str, Any], fix_with_llm: Optional[Callable] = None):
        """
        Initialize a new schema guardrail.
        
        Args:
            name: The name of the guardrail
            description: A description of what the guardrail does
            schema: The JSON schema to validate against
            fix_with_llm: Optional function to fix invalid outputs
        """
        super().__init__(name, description)
        self.schema = schema
        self.fix_with_llm = fix_with_llm
    
    def validate(self, output: str) -> Dict[str, Any]:
        """
        Validate the output against the JSON schema.
        
        Args:
            output: The output to validate
            
        Returns:
            A dictionary containing the validation result
        """
        import json
        from jsonschema import validate, ValidationError
        
        try:
            data = json.loads(output)
            validate(instance=data, schema=self.schema)
            return {
                "valid": True,
                "issues": []
            }
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "issues": [{"message": f"Invalid JSON: {str(e)}"}]
            }
        except ValidationError as e:
            return {
                "valid": False,
                "issues": [{"message": str(e)}]
            }
    
    def fix(self, output: str, validation_result: Dict[str, Any]) -> str:
        """
        Fix the output to conform to the JSON schema.
        
        Args:
            output: The original output
            validation_result: The result of the validation
            
        Returns:
            The fixed output
        """
        if validation_result["valid"]:
            return output
        
        if self.fix_with_llm:
            return self.fix_with_llm(output, validation_result, self.schema)
        
        return output  # Return original if no fix function is provided
