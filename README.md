# Agent Workflow Framework

A framework for agent-based workflow nodes with instructions, tools, and guardrails properties.

## Overview

This framework enhances the traditional workflow node structure by giving nodes agent-like capabilities. Each node can:

- Have specific instructions to guide its behavior
- Use defined tools to interact with external systems
- Apply guardrails for safety and quality control
- Configure context inheritance between nodes

## Key Features

- **Agent-like Nodes**: Nodes function like agents with specific instructions and capabilities
- **Tool Integration**: Easily define and use tools within nodes
- **Guardrails**: Apply safety and quality controls to node outputs
- **Context Control**: Flexible configuration for context inheritance between nodes
- **Unified Interface**: Consistent API across different LLM providers
- **Conditional Workflows**: Support for branching based on node output

## Installation

```bash
pip install agent-workflow-framework
```

## Basic Usage

```python
from agent_workflow_framework import AgentNode, AgentWorkflow, NodeState, LLM

# Define a state class
class MyState(NodeState):
    data: str = ""
    processed_data: str = ""

# Create an agent node
class DataProcessor(AgentNode[MyState]):
    name = "data_processor"
    instruction = "Process the input data by summarizing it."
    
    def proc(self, state: MyState) -> MyState:
        # Process will use the node's instruction with the LLM
        state.processed_data = self.run_with_instruction(state.data)
        return state

# Create a workflow
workflow = AgentWorkflow([DataProcessor(llm)], MyState)
app = workflow.get_app()

# Run the workflow
result = app.invoke({"data": "This is some text that needs processing."})
```

## Advanced Features

### Tools

You can add tools to nodes for external capabilities:

```python
from agent_workflow_framework.tools import JSONSchemaTool

# Define a tool
search_tool = JSONSchemaTool(
    name="search",
    description="Search the web for information",
    function=search_function,
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
)

# Add to a node
class ResearchNode(AgentNode[MyState]):
    name = "researcher"
    instruction = "Research the topic using the search tool."
    
    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.register_tool(search_tool)
    
    def proc(self, state: MyState) -> MyState:
        result = self.run_with_tools(state.topic)
        state.research_results = result["response"]
        return state
```

### Guardrails

Add guardrails to ensure the quality and safety of the output:

```python
from agent_workflow_framework.guardrails import RegexGuardrail

# Define a guardrail
content_filter = RegexGuardrail(
    name="content_filter",
    description="Filter out inappropriate content",
    patterns=[
        {
            "pattern": r"\b(inappropriate|offensive)\s+word\b",
            "action": "block",
            "message": "Inappropriate content detected"
        }
    ]
)

# Add to a node
class ContentCreator(AgentNode[MyState]):
    name = "content_creator"
    instruction = "Create content based on the input."
    
    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.register_guardrail(content_filter)
    
    def proc(self, state: MyState) -> MyState:
        # The guardrail will automatically be applied
        state.content = self.run_with_instruction(state.prompt)
        return state
```

### Context Sharing

Control context inheritance between nodes:

```python
class DataCollector(AgentNode[MyState]):
    name = "data_collector"
    instruction = "Collect data from the input."
    # Share context with other nodes
    share_context = True
    
    def proc(self, state: MyState) -> MyState:
        # Add to context
        state.add_to_context("timestamp", "2025-05-02")
        state.data = self.run_with_instruction(state.input)
        return state

class DataAnalyzer(AgentNode[MyState]):
    name = "data_analyzer"
    instruction = "Analyze the data."
    # Use context from other nodes
    use_context = True
    
    def proc(self, state: MyState) -> MyState:
        # Get from context
        timestamp = state.get_from_context("timestamp", "unknown")
        state.analysis = self.run_with_instruction(
            state.data, 
            {"timestamp": timestamp}
        )
        return state
```

### Conditional Workflows

Create workflows with branches:

```python
from agent_workflow_framework.core.workflow import ConditionalWorkflow

# Define a condition function
def complexity_condition(state: MyState) -> str:
    if state.complexity == "SIMPLE":
        return "simple"
    elif state.complexity == "COMPLEX":
        return "complex"
    else:
        return "error"

# Create nodes
analyzer = ComplexityAnalyzerNode(llm)
simple_processor = SimpleProcessorNode(llm)
complex_processor = ComplexProcessorNode(llm)

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

# Create conditional workflow
workflow = ConditionalWorkflow(
    [analyzer, simple_processor, complex_processor],
    MyState,
    conditional_edges
)
```

## Examples

Check out the examples directory for more detailed examples:

- [Simple Workflow](examples/simple_workflow.py)
- [Advanced Workflow](examples/advanced_workflow.py)
- [Conditional Workflow](examples/conditional_workflow.py)

## License

MIT
