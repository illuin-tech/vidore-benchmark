"""
Utility functions for working with images.
"""

import base64
import io
from pathlib import Path
from typing import cast

import datasets
from datasets import Dataset
from PIL import Image
from torch.utils.data import IterableDataset
from tqdm import tqdm


def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    # Calculate the scaling factor
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image


def scale_to_max_dimension(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
    """
    Scale an image to a maximum dimension while maintaining the aspect ratio.
    """
    # Get the dimensions of the image
    width, height = image.size

    max_original_dimension = max(width, height)

    if max_original_dimension < max_dimension:
        return image

    # Calculate the scaling factor
    aspect_ratio = max_dimension / max_original_dimension
    new_width = int(width * aspect_ratio)
    new_height = int(height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image


def get_base64_image(img: str | Image.Image, add_url_prefix: bool = True) -> str:
    """
    Convert an image (from a filepath or a PIL.Image object) to a JPEG-base64 string.
    """
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise ValueError("`img` must be a path to an image or a PIL Image object.")

    buffered = io.BytesIO()
    img.save(buffered, format="jpeg")
    b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_data}" if add_url_prefix else b64_data


def shorten_image_path(image_path: str) -> str:
    """
    Shorten the image path to make it more readable.
    """
    full_path = Path(image_path)
    enclosing_dir = full_path.parent.parent
    return full_path.relative_to(enclosing_dir).as_posix()


def generate_dataset_from_img_folder(path_to_folder: str) -> IterableDataset:
    """
    Generate a dataset from a folder containing JPG images.

    Args:
    - path_to_folder (str): path to the folder containing the pdf files

    Returns:
    - ds (DatasetDict): a dataset containing the questions and answers generated from the pdf files

    structure of the dataset:
    - query (str): the question generated from the image
    - image (PIL.Image): the image
    - image_filename (str): the path to the image

    """
    img_files = list(Path(path_to_folder).rglob("*.jpg")) + list(Path(path_to_folder).rglob("*.jpeg"))

    # Create a Dataset from the dictionary
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "image_filename": datasets.Value("string"),
        }
    )

    def gen():
        with tqdm(total=len(img_files)) as pbar:
            for image_path in img_files:
                pbar.set_description(f"Processing {shorten_image_path(str(image_path))}")

                pil_image = Image.open(image_path)

                yield {
                    "image": pil_image,
                    "image_filename": image_path,
                }
                pbar.update(1)

    # Create the dataset from the generator
    ds = cast(IterableDataset, Dataset.from_generator(gen, features=features))

    return ds
