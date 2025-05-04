"""
Agent node module for agent workflow framework.

This module provides the AgentNode class that implements agent-like node behavior.
"""

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from agent_workflow_framework.core.state import NodeState
from agent_workflow_framework.guardrails.registry import GuardrailRegistry
from agent_workflow_framework.llm.base import LLM
from agent_workflow_framework.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=NodeState)


class AgentNode(Generic[T]):
    """
    Base class for agent-like workflow nodes.

    This class provides agent-like behavior for workflow nodes, including:
    - Customizable instructions
    - Tool integration
    - Guardrails for safety and quality control
    - Context control
    """

    # Class-level attributes that can be overridden by subclasses
    name: str = "unnamed_node"
    instruction: str = ""
    use_context: bool = True
    share_context: bool = True

    def __init__(self, llm: LLM):
        """
        Initialize a new agent node.

        Args:
            llm: The language model to use for the node
        """
        self.llm = llm
        self.tools = ToolRegistry()
        self.guardrails = GuardrailRegistry()

    def action(self, state: T) -> T:
        """
        Execute the node's action on the state.

        This method handles error management and context propagation.

        Args:
            state: The input state

        Returns:
            The updated state
        """
        try:
            self.validate(state)
            logger.info(f"Node '{self.name}' starting")

            # Apply the agent node's processing
            updated_state = self.proc(state)

            logger.info(f"Node '{self.name}' completed")
            return updated_state
        except Exception as e:
            logger.error(f"Error in node '{self.name}': {str(e)}")
            return state.emit_error(
                f"An error occurred in node '{self.name}': {str(e)}"
            )

    @abstractmethod
    def proc(self, state: T) -> T:
        """
        Process the state with the agent node's logic.

        This method should be implemented by subclasses to define the node's behavior.

        Args:
            state: The input state

        Returns:
            The updated state
        """
        pass

    def validate(self, state: T) -> None:
        """
        Validate the input state for the node.

        This method can be overridden by subclasses to implement custom validation.

        Args:
            state: The input state

        Raises:
            Exception: If the validation fails
        """
        pass

    def run_with_instruction(
        self, input_text: str, additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Run the language model with the node's instruction.

        Args:
            input_text: The input text to process
            additional_context: Additional context for the prompt

        Returns:
            The model's response
        """
        # Construct the prompt
        prompt = self._build_prompt(input_text, additional_context)

        # Run the language model
        response = self.llm.invoke(prompt)

        # Apply guardrails
        safe_response = self.guardrails.apply_all(response)

        return safe_response

    def run_with_tools(
        self, input_text: str, additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the language model with the node's instruction and tools.

        Args:
            input_text: The input text to process
            additional_context: Additional context for the prompt

        Returns:
            The model's response with tool calls
        """
        # Construct the prompt
        prompt = self._build_prompt(input_text, additional_context)

        # Convert tools to format expected by LLM
        tools_list = self.tools.to_dict_list()

        # Run the language model with tools
        return self.llm.invoke_with_tools(prompt, tools_list)

    def _build_prompt(
        self, input_text: str, additional_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build the prompt for the language model.

        Args:
            input_text: The input text to process
            additional_context: Additional context for the prompt

        Returns:
            The constructed prompt
        """
        instruction = self.instruction.strip()

        # Add additional context if provided
        context_str = ""
        if additional_context:
            context_str = "\nContext:\n"
            for key, value in additional_context.items():
                context_str += f"{key}: {value}\n"

        return f"{instruction}\n{context_str}\nInput:\n{input_text}\n"

    def register_tool(self, tool) -> None:
        """
        Register a tool with the node.

        Args:
            tool: The tool to register
        """
        self.tools.register(tool)

    def register_guardrail(self, guardrail) -> None:
        """
        Register a guardrail with the node.

        Args:
            guardrail: The guardrail to register
        """
        self.guardrails.register(guardrail)

    def generate_node(self) -> Tuple[str, Callable[[T], T]]:
        """
        Generate a node for use with langgraph.

        Returns:
            A tuple of (node_name, action_function)
        """
        return self.node_name, self.action

    @property
    def node_name(self) -> str:
        """
        Get the node name in a format suitable for langgraph.

        Returns:
            The node name
        """
        return self.name.replace(" ", "_")


class StructuredOutputNode(AgentNode[T]):
    """
    Node that produces structured output according to a schema.
    """

    def __init__(self, llm: LLM, output_schema: Dict[str, Any]):
        """
        Initialize a new structured output node.

        Args:
            llm: The language model to use
            output_schema: JSON schema for the expected output
        """
        super().__init__(llm)
        self.output_schema = output_schema

    def run_with_schema(
        self, input_text: str, additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the language model with the node's instruction and output schema.

        Args:
            input_text: The input text to process
            additional_context: Additional context for the prompt

        Returns:
            The model's structured response
        """
        # Construct the prompt
        prompt = self._build_prompt(input_text, additional_context)

        # Run the language model with the schema
        return self.llm.invoke_with_structured_output(prompt, self.output_schema)


class MultiStepNode(AgentNode[T]):
    """
    Node that performs multiple steps of processing.
    """

    def __init__(self, llm: LLM, steps: List[Tuple[str, Callable]]):
        """
        Initialize a new multi-step node.

        Args:
            llm: The language model to use
            steps: List of (step_name, step_function) tuples
        """
        super().__init__(llm)
        self.steps = steps

    def proc(self, state: T) -> T:
        """
        Process the state with multiple steps.

        Args:
            state: The input state

        Returns:
            The updated state
        """
        current_state = state

        for step_name, step_function in self.steps:
            logger.info(f"Running step '{step_name}' in node '{self.name}'")
            try:
                current_state = step_function(current_state)
            except Exception as e:
                logger.error(
                    f"Error in step '{step_name}' of node '{self.name}': {str(e)}"
                )
                return current_state.emit_error(
                    f"Error in step '{step_name}': {str(e)}"
                )

        return current_state
