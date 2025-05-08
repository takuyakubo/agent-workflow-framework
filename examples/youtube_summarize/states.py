from agent_workflow_framework import NodeState


class YoutubeSummarizeState(NodeState):
    url: str  # url
    summary: str = ""  # 要約


input_keys = ["url"]
output_key = "summary"
