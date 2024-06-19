from __future__ import annotations

from dataclasses import dataclass
from typing import List, cast

import torch
from PIL import Image
from transformers import LlamaTokenizerFast, PaliGemmaProcessor


@dataclass
class ColPaliTextInput:
    """
    Dataclass for text inputs to ColPali.

    Usage:
    >>> from dataclasses import asdict
    >>> model = ColPaliProcessor.from_pretrained("my_model_name")
    >>> processor = ColPaliProcessor.from_pretrained("my_model_name")
    >>> input_text_processed = processor.process_text("The quick brown fox jumps over the lazy dog.").to("cuda:0")
    >>> with torch.no_grad():
    >>>     output_text = model.forward(**asdict(input_text_processed))
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device) -> ColPaliTextInput:
        return ColPaliTextInput(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )


@dataclass
class ColPaliImageInput:
    """
    Dataclass for image inputs to ColPali.

    Usage:
    >>> from dataclasses import asdict
    >>> model = ColPaliProcessor.from_pretrained("my_model_name")
    >>> processor = ColPaliProcessor.from_pretrained("my_model_name")
    >>> input_image_processed = processor.process_image("The quick brown fox jumps over the lazy dog.").to("cuda:0")
    >>> with torch.no_grad():
    >>>     output_image = model.forward(**asdict(input_image_processed))
    """

    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: str | torch.device) -> ColPaliImageInput:
        return ColPaliImageInput(
            input_ids=self.input_ids.to(device),
            pixel_values=self.pixel_values.to(device),
            attention_mask=self.attention_mask.to(device),
        )


class ColPaliProcessor:
    """
    Wrapper class for the PaliGemmaProcessor with additional methods for processing text and image inputs for ColPali.
    """

    def __init__(self, processor: PaliGemmaProcessor):
        self.processor = processor
        self.tokenizer = cast(LlamaTokenizerFast, self.processor.tokenizer)  # type: ignore

    @staticmethod
    def from_pretrained(model_name: str) -> ColPaliProcessor:
        return ColPaliProcessor(processor=cast(PaliGemmaProcessor, PaliGemmaProcessor.from_pretrained(model_name)))

    def process_text(
        self,
        text: str | List[str],
        padding: str = "longest",
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> ColPaliTextInput:
        """
        Process text inputs for the model.
        If `add_special_tokens` is True (default), the text will be prepended with the <bos>
        token and appended with " \n".
        """
        if add_special_tokens:
            if isinstance(text, str):
                text = self.tokenizer.bos_token + text + "\n"
            elif isinstance(text, list):
                text = [self.tokenizer.bos_token + t + "\n" for t in text]
            else:
                raise ValueError("text must be a string or a list of strings.")

        batch_output = self.tokenizer(
            text, padding=padding, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )

        return ColPaliTextInput(
            input_ids=cast(torch.Tensor, batch_output["input_ids"]),
            attention_mask=cast(torch.Tensor, batch_output["attention_mask"]),
        )

    def process_image(
        self,
        image: Image.Image | List[Image.Image],
        padding: str = "longest",
        do_convert_rgb: bool = True,
        return_tensors: str = "pt",
        add_special_prompt: bool = True,
    ) -> ColPaliImageInput:
        """
        Process image inputs for the model.

        If `add_special_prompt` is True (default), the image will be prepended with the special
        prompt "Describe the image." (which was used during training).
        """

        special_prompt = "Describe the image." if add_special_prompt else None
        if isinstance(image, Image.Image):
            text_input = [special_prompt]
        elif isinstance(image, list):
            text_input = [special_prompt] * len(image)
        else:
            raise ValueError("image must be a PIL Image or a list of PIL Images.")

        batch_output = self.processor(
            text=text_input,
            images=image,
            padding=padding,
            do_convert_rgb=do_convert_rgb,
            return_tensors=return_tensors,
        )

        if add_special_prompt:
            return ColPaliImageInput(
                input_ids=batch_output["input_ids"],
                pixel_values=batch_output["pixel_values"],
                attention_mask=batch_output["attention_mask"],
            )
        else:
            return ColPaliImageInput(
                input_ids=batch_output["input_ids"][:, : self.processor.image_seq_length],
                pixel_values=batch_output["pixel_values"][:, : self.processor.image_seq_length],
                attention_mask=batch_output["attention_mask"][:, : self.processor.image_seq_length],
            )

    def decode(self, *args, **kwargs):
        """
        Call the tokenizer's decode method.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        Call the tokenizer's batch_decode method.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
