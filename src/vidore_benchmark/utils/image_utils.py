import hashlib

from PIL import Image


def hash_image(image: Image.Image) -> str:
    """
    Hash a PILLOW image using MD5.

    Args:
        image (Image.Image): PIL Image object.

    Returns:
        str: MD5 hash of the image.
    """
    return hashlib.md5(image.tobytes()).hexdigest()
