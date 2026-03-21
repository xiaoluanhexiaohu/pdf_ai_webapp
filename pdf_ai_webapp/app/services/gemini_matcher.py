from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

from app.schemas.placement import PlacementDecision


class GeminiMatcher:
    def __init__(self, api_key: str = "", model: str = "gemini-2.5-flash", training_rules: List[Dict[str, Any]] | None = None):
        self.api_key = api_key.strip()
        self.model = model
        self.training_rules = training_rules or []
        self.client = None

        if self.api_key and genai is not None:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)

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
                    reason="未配置 Gemini 或 Gemini 输出失败，按顺序把图片依次匹配到锚点。",
                )
            )
        return placements

    def _rule_based_match(self, images: List[Dict[str, Any]], anchors: List[Dict[str, Any]]) -> tuple[List[PlacementDecision], List[Dict[str, Any]]]:
        placements: List[PlacementDecision] = []
        remaining_images: List[Dict[str, Any]] = []

        for image in images:
            filename = image["filename"].lower()
            matched_rule = None
            for rule in self.training_rules:
                keyword = str(rule.get("keyword", "")).strip().lower()
                if keyword and keyword in filename:
                    matched_rule = rule
                    break

            if not matched_rule:
                remaining_images.append(image)
                continue

            anchor = next((a for a in anchors if a["anchor_text"] == matched_rule.get("anchor_text")), None)
            if not anchor:
                remaining_images.append(image)
                continue

            placements.append(
                PlacementDecision(
                    image_id=image["image_id"],
                    filename=image["filename"],
                    anchor_id=anchor["anchor_id"],
                    page_number=anchor["page_number"],
                    mode=matched_rule.get("mode", "below"),
                    size_hint="medium",
                    confidence=0.90,
                    reason=f"命中训练规则：{matched_rule.get('keyword')} -> {matched_rule.get('anchor_text')}",
                )
            )

        return placements, remaining_images

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)

    def match(self, images: List[Dict[str, Any]], anchors: List[Dict[str, Any]], page_preview_map: Dict[int, str]) -> List[PlacementDecision]:
        trained_placements, remaining_images = self._rule_based_match(images, anchors)
        if not remaining_images:
            return trained_placements

        if not self.client:
            return trained_placements + self._heuristic_match(remaining_images, anchors)

        try:
            payload = {
                "anchors": anchors,
                "images": [{k: v for k, v in item.items() if k != "path"} for item in remaining_images],
                "training_rules": self.training_rules[:30],
                "rules": {
                    "modes": ["below", "right", "appendix_page"],
                    "notes": [
                        "如果页面空间较少，优先 appendix_page。",
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
                            "reason": "why",
                        }
                    ]
                },
            }

            prompt = (
                "你是一个文档插图助手。请根据锚点和图片内容为每张图片匹配最合适锚点与插图模式。"
                "只返回 JSON，不要输出任何额外解释。\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            )

            contents: List[Any] = [prompt]
            for image in remaining_images:
                mime = mimetypes.guess_type(image["path"])[0] or "image/png"
                with open(image["path"], "rb") as f:
                    contents.append({"mime_type": mime, "data": f.read()})
            for preview_path in page_preview_map.values():
                with open(preview_path, "rb") as f:
                    contents.append({"mime_type": "image/png", "data": f.read()})

            response = self.client.generate_content(contents)
            text = response.text or ""
            data = self._extract_json(text)
            placements = [PlacementDecision(**item) for item in data.get("placements", [])]
            if not placements:
                placements = self._heuristic_match(remaining_images, anchors)
            return trained_placements + placements
        except Exception:
            return trained_placements + self._heuristic_match(remaining_images, anchors)
