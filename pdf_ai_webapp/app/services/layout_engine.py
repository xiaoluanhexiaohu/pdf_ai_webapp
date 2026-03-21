from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import fitz
from PIL import Image


@dataclass
class LayoutConfig:
    max_image_width_ratio: float = 0.42
    max_image_height_ratio: float = 0.28
    gap: int = 12
    margin: int = 24


def rects_intersect(a: fitz.Rect, b: fitz.Rect) -> bool:
    return not (a.x1 <= b.x0 or a.x0 >= b.x1 or a.y1 <= b.y0 or a.y0 >= b.y1)


class LayoutEngine:
    def __init__(self, config: LayoutConfig):
        self.config = config

    @staticmethod
    def _scale_image(img_w: int, img_h: int, max_w: float, max_h: float) -> Tuple[float, float]:
        ratio = min(max_w / img_w, max_h / img_h)
        ratio = min(ratio, 1.0)
        return img_w * ratio, img_h * ratio

    def _occupied(self, occupied_rects: List[List[float]]) -> List[fitz.Rect]:
        return [fitz.Rect(*r) for r in occupied_rects]

    def _fits(self, candidate: fitz.Rect, occupied: List[fitz.Rect], page_rect: fitz.Rect) -> bool:
        if candidate.x0 < self.config.margin or candidate.y0 < self.config.margin:
            return False
        if candidate.x1 > page_rect.x1 - self.config.margin:
            return False
        if candidate.y1 > page_rect.y1 - self.config.margin:
            return False
        return all(not rects_intersect(candidate, occ) for occ in occupied)

    def choose_rect(
        self,
        image_path: str,
        page_width: float,
        page_height: float,
        anchor_rect: List[float],
        occupied_rects: List[List[float]],
        preferred_mode: str,
    ) -> Tuple[str, List[float] | None]:
        page_rect = fitz.Rect(0, 0, page_width, page_height)
        anchor = fitz.Rect(*anchor_rect)
        occupied = self._occupied(occupied_rects)

        with Image.open(image_path) as image:
            img_w, img_h = image.width, image.height

        max_w = page_width * self.config.max_image_width_ratio
        max_h = page_height * self.config.max_image_height_ratio
        draw_w, draw_h = self._scale_image(img_w, img_h, max_w, max_h)

        candidates: List[Tuple[str, fitz.Rect]] = []
        below = fitz.Rect(anchor.x0, anchor.y1 + self.config.gap, anchor.x0 + draw_w, anchor.y1 + self.config.gap + draw_h)
        right = fitz.Rect(anchor.x1 + self.config.gap, anchor.y0, anchor.x1 + self.config.gap + draw_w, anchor.y0 + draw_h)

        mode_order = [preferred_mode, "below", "right", "appendix_page"]
        unique_modes = []
        for mode in mode_order:
            if mode not in unique_modes:
                unique_modes.append(mode)

        for mode in unique_modes:
            if mode == "below":
                candidates.append(("below", below))
            elif mode == "right":
                candidates.append(("right", right))

        for mode, candidate in candidates:
            if self._fits(candidate, occupied, page_rect):
                return mode, [candidate.x0, candidate.y0, candidate.x1, candidate.y1]

        return "appendix_page", None
