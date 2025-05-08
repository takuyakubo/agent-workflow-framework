from typing import Any, Dict, List

from pydantic import Field

from agent_workflow_framework import NodeState


class ImageProcState(NodeState):
    images: List[Any] = Field(default=[])  # 画像のリスト
    image_content: List[Dict[str, Any]] = Field(default=[])  # 画像分析結果


input_keys = ["images"]
output_key = "image_content"
