from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BoundingBox:
    """
    Bounding box data structure where (x0, y0) are the coordinates of the top-left corner.
    The x-axis is pointing to the right and the y-axis is pointing down. The pixel
    coordinates are absolute and stored as integers.
    """

    x0: int
    x1: int
    y0: int
    y1: int

    def __lt__(self, other: BoundingBox) -> bool:
        """
        Implements the lexical order (left-to-right, top-to-bottom).
        """
        if self.y0 < other.y0:
            return True
        elif self.y0 == other.y0:
            if self.x0 < other.x0:
                return True
            else:
                return False
        return False

    def __le__(self, other: BoundingBox) -> bool:
        return self < other or self == other

    @property
    def area(self) -> int:
        return (self.y1 - self.y0) * (self.x1 - self.x0)

    @property
    def height(self) -> int:
        """
        Returns the height of the box.
        """
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        """
        Returns the width of the box.
        """
        return self.x1 - self.x0

    @property
    def center(self) -> Tuple[float, float]:
        """
        Returns the center coordinates of the box.
        """
        x_center = 0.5 * (self.x0 + self.x1)
        y_center = 0.5 * (self.y0 + self.y1)
        return x_center, y_center

    def is_in(self, other_box: BoundingBox) -> bool:
        return (
            self.x0 >= other_box.x0 and self.x1 <= other_box.x1 and self.y0 >= other_box.y0 and self.y1 <= other_box.y1
        )


@dataclass
class ExtractedWord:
    """
    Contains one word from the output of an OCR model (e.g. Tesseract).
    """

    text: str
    bbox: BoundingBox
    extraction_confidence: float

    def __lt__(self, other: ExtractedWord) -> bool:
        return self.bbox < other.bbox

    def __le__(self, other: ExtractedWord) -> bool:
        return self.bbox <= other.bbox
