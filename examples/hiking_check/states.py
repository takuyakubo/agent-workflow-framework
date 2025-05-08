from typing import Any, Dict, List

from pydantic import Field

from agent_workflow_framework import NodeState


class HikingCheckState(NodeState):
    city: str  # 都市名
    check_result: str = ""  # チェック結果


input_keys = ["city"]
output_key = "check_result"
