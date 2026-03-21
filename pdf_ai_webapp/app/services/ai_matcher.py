from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from app.schemas.placement import PlacementDecision


class AIMatcher:
    def __init__(self, api_key: str = "", model: str = "gpt-4.1-mini"):
        self.api_key = api_key.strip()
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if (self.api_key and OpenAI is not None) else None

    @staticmethod
    def _to_base64(path: str | Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _heuristic_match(images: List[Dict[str, Any]], anchors: List[Dict[str, Any]]) -> List[PlacementDecision]:
        placements: List[PlacementDecision] = []
        if not anchors:
            return placements
        for index, image in enumerate(images):
            anchor = anchors[index % len(anchors)]
            placements.append(
                PlacementDecision(
                    image_id=image["image_id"],
                    filename=image["filename"],
                    anchor_id=anchor["anchor_id"],
                    page_number=anchor["page_number"],
                    mode="below",
                    size_hint="medium",
                    confidence=0.45,
                    reason="未配置 AI 或 AI 输出失败，按顺序把图片依次匹配到锚点。",
                )
            )
        return placements

    def match(self, images: List[Dict[str, Any]], anchors: List[Dict[str, Any]], page_preview_map: Dict[int, str]) -> List[PlacementDecision]:
        if not self.client:
            return self._heuristic_match(images, anchors)

        try:
            input_parts: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "你是一个 PDF 智能排版助手。任务：根据图片内容、锚点文字、页面截图，"
                                "给每张图片选择最合适的锚点，并仅在 below/right/appendix_page 三种模式中选择一种。"
                                "必须只返回 JSON，不能输出额外解释。"
                            ),
                        }
                    ],
                }
            ]

            payload = {
                "anchors": anchors,
                "images": [{k: v for k, v in item.items() if k != "path"} for item in images],
                "rules": {
                    "modes": ["below", "right", "appendix_page"],
                    "notes": [
                        "如果页面现有空间较少，优先选择 appendix_page。",
                        "如果锚点右侧明显更自然，可选择 right。",
                        "常规情况优先 below。",
                    ],
                },
                "expected_output": {
                    "placements": [
                        {
                            "image_id": "string",
                            "filename": "string",
                            "anchor_id": "string",
                            "page_number": 1,
                            "mode": "below|right|appendix_page",
                            "size_hint": "small|medium|large",
                            "confidence": 0.95,
                            "reason": "why"
                        }
                    ]
                },
            }
            input_parts.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
                }
            )

            for image in images:
                b64 = self._to_base64(image["path"])
                input_parts[1]["content"].append(
                    {
                        "type": "input_text",
                        "text": f"下面是一张用户上传图片，image_id={image['image_id']}，filename={image['filename']}。",
                    }
                )
                input_parts[1]["content"].append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                    }
                )

            for page_number, preview_path in page_preview_map.items():
                b64 = self._to_base64(preview_path)
                input_parts[1]["content"].append(
                    {
                        "type": "input_text",
                        "text": f"下面是 PDF 第 {page_number} 页截图。",
                    }
                )
                input_parts[1]["content"].append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                    }
                )

            response = self.client.responses.create(
                model=self.model,
                input=input_parts,
                max_output_tokens=1800,
            )
            text = getattr(response, "output_text", "") or ""
            data = json.loads(text)
            placements = [PlacementDecision(**item) for item in data.get("placements", [])]
            if not placements:
                return self._heuristic_match(images, anchors)
            return placements
        except Exception:
            return self._heuristic_match(images, anchors)
