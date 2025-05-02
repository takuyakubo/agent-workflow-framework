"""
Workflow module for agent workflow framework.

This module provides the AgentWorkflow class for creating and managing agent workflows.
"""

import logging
from typing import Dict, List, Type

from langgraph.graph import END, START, StateGraph

from agent_workflow_framework.core.node import AgentNode
from agent_workflow_framework.core.state import NodeState

logger = logging.getLogger(__name__)


class AgentWorkflow:
    """
    Workflow for connecting agent nodes.
    
    This class provides methods for creating and managing workflows of agent nodes.
    """
    
    def __init__(self, nodes: List[AgentNode], state_class: Type[NodeState]):
        """
        Initialize a new workflow.
        
        Args:
            nodes: The nodes in the workflow
            state_class: The state class to use for the workflow
        """
        self.nodes = nodes
        self.state_class = state_class
        self.workflow = StateGraph(state_class)
        self._setup()
    
    def _setup(self) -> None:
        """
        Set up the workflow by adding nodes and edges.
        """
        # Add all nodes to the workflow
        for node in self.nodes:
            self.workflow.add_node(node.node_name, node.action)
        
        # Set the first node as the entry point
        if self.nodes:
            self.workflow.set_entry_point(self.nodes[0].node_name)
        
        # Connect the nodes in sequence
        for i in range(len(self.nodes) - 1):
            self.workflow.add_conditional_edges(
                self.nodes[i].node_name,
                self._check_error,
                {"error": END, "continue": self.nodes[i + 1].node_name},
            )
        
        # Connect the last node to the end
        if self.nodes:
            self.workflow.add_edge(self.nodes[-1].node_name, END)
    
    @staticmethod
    def _check_error(state: NodeState) -> str:
        """
        Check if the state has an error and decide the next step.
        
        Args:
            state: The current state
            
        Returns:
            "error" if the state has an error, "continue" otherwise
        """
        if state.error:
            logger.error(f"Workflow error: {state.error}")
            return "error"
        return "continue"
    
    def get_app(self):
        """
        Get the compiled workflow application.
        
        Returns:
            The compiled workflow application
        """
        return self.workflow.compile()
    
    def add_node(self, node: AgentNode, after: str = None) -> None:
        """
        Add a node to the workflow.
        
        Args:
            node: The node to add
            after: The name of the node to add this node after (optional)
            
        Raises:
            ValueError: If the node name already exists or the after node does not exist
        """
        # Check if the node already exists
        if any(n.node_name == node.node_name for n in self.nodes):
            raise ValueError(f"Node with name '{node.node_name}' already exists")
        
        # If after is specified, find the index to insert the node
        if after:
            after_index = next((i for i, n in enumerate(self.nodes) if n.node_name == after), None)
            if after_index is None:
                raise ValueError(f"Node with name '{after}' does not exist")
            
            self.nodes.insert(after_index + 1, node)
        else:
            # Otherwise, add the node to the end
            self.nodes.append(node)
        
        # Rebuild the workflow
        self.workflow = StateGraph(self.state_class)
        self._setup()


class ConditionalWorkflow(AgentWorkflow):
    """
    Workflow that supports conditional branching.
    
    This class extends AgentWorkflow to support conditional branching between nodes.
    """
    
    def __init__(self, nodes: List[AgentNode], state_class: Type[NodeState], conditional_edges: Dict = None):
        """
        Initialize a new conditional workflow.
        
        Args:
            nodes: The nodes in the workflow
            state_class: The state class to use for the workflow
            conditional_edges: Dictionary of conditional edges, where keys are node names and
                values are dictionaries mapping conditions to target node names
        """
        self.conditional_edges = conditional_edges or {}
        super().__init__(nodes, state_class)
    
    def _setup(self) -> None:
        """
        Set up the workflow by adding nodes and conditional edges.
        """
        # Add all nodes to the workflow
        for node in self.nodes:
            self.workflow.add_node(node.node_name, node.action)
        
        # Set the first node as the entry point
        if self.nodes:
            self.workflow.set_entry_point(self.nodes[0].node_name)
        
        # Add conditional edges
        for node_name, edges in self.conditional_edges.items():
            condition_function = edges.get("condition")
            destinations = edges.get("destinations", {})
            
            if condition_function and destinations:
                self.workflow.add_conditional_edges(
                    node_name,
                    condition_function,
                    destinations,
                )
            elif not condition_function and not destinations:
                # If no conditional edges, find the next node
                try:
                    node_index = next(i for i, n in enumerate(self.nodes) if n.node_name == node_name)
                    if node_index < len(self.nodes) - 1:
                        next_node = self.nodes[node_index + 1]
                        self.workflow.add_conditional_edges(
                            node_name,
                            self._check_error,
                            {"error": END, "continue": next_node.node_name},
                        )
                    else:
                        self.workflow.add_edge(node_name, END)
                except StopIteration:
                    logger.warning(f"Node '{node_name}' not found in nodes list")
            else:
                logger.warning(f"Invalid conditional edge specification for node '{node_name}'")
        
        # Connect any nodes not covered by conditional edges
        for i, node in enumerate(self.nodes):
            if node.node_name not in self.conditional_edges:
                if i < len(self.nodes) - 1:
                    self.workflow.add_conditional_edges(
                        node.node_name,
                        self._check_error,
                        {"error": END, "continue": self.nodes[i + 1].node_name},
                    )
                else:
                    self.workflow.add_edge(node.node_name, END)
    
    def add_conditional_edge(self, source: str, condition, destinations: Dict[str, str]) -> None:
        """
        Add a conditional edge to the workflow.
        
        Args:
            source: The name of the source node
            condition: The condition function
            destinations: Dictionary mapping conditions to target node names
            
        Raises:
            ValueError: If the source node does not exist
        """
        if not any(n.node_name == source for n in self.nodes):
            raise ValueError(f"Source node '{source}' does not exist")
        
        self.conditional_edges[source] = {
            "condition": condition,
            "destinations": destinations,
        }
        
        # Rebuild the workflow
        self.workflow = StateGraph(self.state_class)
        self._setup()
