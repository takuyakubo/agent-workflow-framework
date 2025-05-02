"""
Conditional workflow example for agent workflow framework.

This example demonstrates a workflow with conditional branching:
1. Content analyzer
2. Conditional branching based on complexity
   a. Simple content processor
   b. Complex content processor
3. Final formatter
"""

import logging
import sys
from typing import Dict, List, Optional

from agent_workflow_framework import AgentNode, LLM, NodeState
from agent_workflow_framework.core.workflow import ConditionalWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# Define a simple mock LLM for testing
class MockLLM(LLM):
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
    
    @property
    def model_name(self) -> str:
        return "mock-llm"
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    def invoke(self, prompt: str, **kwargs) -> str:
        # For simplicity, just return a hardcoded response based on the prompt
        if "analyze the complexity" in prompt:
            if "quantum computing" in prompt.lower():
                return "COMPLEX"
            else:
                return "SIMPLE"
        elif "process simple content" in prompt:
            return "This is a simple explanation of the topic."
        elif "process complex content" in prompt:
            return "This is a detailed analysis of the complex topic with technical details."
        elif "format" in prompt:
            if "simple" in prompt:
                return "# Simple Topic\n\nThis is a simple explanation formatted nicely."
            else:
                return "# Complex Topic\n\nThis is a complex explanation formatted with technical details and diagrams."
        else:
            return "Default response"
    
    def stream(self, prompt: str, **kwargs):
        yield self.invoke(prompt, **kwargs)
    
    def invoke_with_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self.invoke(messages[-1]["content"], **kwargs)


# Define state
class ContentState(NodeState):
    """State for the content processing workflow."""
    content: str = ""
    complexity: str = ""  # "SIMPLE" or "COMPLEX"
    processed_content: str = ""
    formatted_output: str = ""


# Define condition function
def complexity_condition(state: ContentState) -> str:
    """
    Determine the next step based on content complexity.
    
    Args:
        state: The current state
        
    Returns:
        "simple", "complex", or "error"
    """
    if state.error:
        return "error"
    
    if state.complexity == "SIMPLE":
        return "simple"
    elif state.complexity == "COMPLEX":
        return "complex"
    else:
        return "error"


# Define nodes
class ContentAnalyzerNode(AgentNode[ContentState]):
    """Analyze content complexity."""
    name = "content_analyzer"
    instruction = "Analyze the complexity of the following content. Respond with either 'SIMPLE' or 'COMPLEX'."
    
    def proc(self, state: ContentState) -> ContentState:
        complexity = self.run_with_instruction(state.content)
        state.complexity = complexity.strip().upper()
        return state


class SimpleContentProcessorNode(AgentNode[ContentState]):
    """Process simple content."""
    name = "simple_processor"
    instruction = "Process simple content. Provide a clear and straightforward explanation."
    
    def proc(self, state: ContentState) -> ContentState:
        processed = self.run_with_instruction(state.content)
        state.processed_content = processed
        return state


class ComplexContentProcessorNode(AgentNode[ContentState]):
    """Process complex content with detailed analysis."""
    name = "complex_processor"
    instruction = """
    Process complex content with detailed analysis.
    Include technical details and in-depth explanations.
    Consider edge cases and provide comprehensive coverage.
    """
    
    def proc(self, state: ContentState) -> ContentState:
        processed = self.run_with_instruction(state.content)
        state.processed_content = processed
        return state


class FormatterNode(AgentNode[ContentState]):
    """Format the processed content into the final output."""
    name = "formatter"
    instruction = "Format the processed content into a well-structured document with appropriate headings and sections."
    
    def proc(self, state: ContentState) -> ContentState:
        formatted = self.run_with_instruction(
            f"Please format the following {state.complexity.lower()} content:\n\n{state.processed_content}"
        )
        state.formatted_output = formatted
        return state


def main():
    # Create a mock LLM
    llm = MockLLM()
    
    # Create nodes
    analyzer = ContentAnalyzerNode(llm)
    simple_processor = SimpleContentProcessorNode(llm)
    complex_processor = ComplexContentProcessorNode(llm)
    formatter = FormatterNode(llm)
    
    # Define conditional edges
    conditional_edges = {
        analyzer.node_name: {
            "condition": complexity_condition,
            "destinations": {
                "simple": simple_processor.node_name,
                "complex": complex_processor.node_name,
                "error": "end"
            }
        }
    }
    
    # Create workflow with four nodes
    nodes = [analyzer, simple_processor, complex_processor, formatter]
    workflow = ConditionalWorkflow(nodes, ContentState, conditional_edges)
    
    # Add edges from processors to formatter
    workflow.add_conditional_edge(
        simple_processor.node_name,
        workflow._check_error,
        {"error": "end", "continue": formatter.node_name}
    )
    workflow.add_conditional_edge(
        complex_processor.node_name,
        workflow._check_error,
        {"error": "end", "continue": formatter.node_name}
    )
    
    # Compile the workflow
    app = workflow.get_app()
    
    # Run workflow with simple content
    simple_result = app.invoke({
        "content": "The earth revolves around the sun in an elliptical orbit."
    })
    
    # Run workflow with complex content
    complex_result = app.invoke({
        "content": "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computation."
    })
    
    # Print the results
    print("\n=== Simple Content Result ===")
    print(f"Complexity: {simple_result['complexity']}")
    print(f"Processed by: Simple Processor")
    print(f"Formatted output:\n{simple_result['formatted_output']}")
    
    print("\n=== Complex Content Result ===")
    print(f"Complexity: {complex_result['complexity']}")
    print(f"Processed by: Complex Processor")
    print(f"Formatted output:\n{complex_result['formatted_output']}")


if __name__ == "__main__":
    main()
