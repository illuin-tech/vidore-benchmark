from typing import Any, Dict, List

from vidore_benchmark.ocr.interfaces import BoundingBox, ExtractedWord


def to_extracted_words(data: Dict[str, Any]) -> List[ExtractedWord]:
    """
    Util function to convert Tesseract output to ExtractedWord objects.
    """
    words: List[ExtractedWord] = []

    for text, x0, y0, word_w, word_h, ocr_conf in zip(
        data["text"], data["left"], data["top"], data["width"], data["height"], data["conf"]
    ):
        if text.strip():
            words.append(
                ExtractedWord(
                    text=text,
                    bbox=BoundingBox(
                        x0=x0,
                        y0=y0,
                        x1=(x0 + word_w),
                        y1=(y0 + word_h),
                    ),
                    extraction_confidence=ocr_conf,
                )
            )
    return words
