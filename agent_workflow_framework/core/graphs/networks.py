from typing import List

from core.graphs.elements import AgentNode, LangGraphConditionalEdge
from langgraph.graph import END, START, StateGraph


class SequentialWorkflow:
    def __init__(self, nodes: List[AgentNode], init_state_cls) -> None:
        self.workflow = StateGraph(init_state_cls)
        self.setup(nodes)

    def setup(self, nodes: List[AgentNode]) -> None:
        nodes_with_sentinels = [START] + nodes + [END]
        edges = [
            LangGraphConditionalEdge(s, t)
            for s, t in zip(nodes_with_sentinels, nodes_with_sentinels[1:])
        ]
        for e in edges:
            if e.target == END:
                self.workflow.add_edge(e.source.node_name, END)
                continue
            self.workflow.add_node(*e.target.generate_node())
            if e.source == START:
                self.workflow.set_entry_point(e.target.node_name)
            else:
                self.workflow.add_conditional_edges(*e.args_conditional_edge())

    def get_app(self):
        return self.workflow.compile()
