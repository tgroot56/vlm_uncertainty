"""Visual perturbation functions for the activation-patching experiment."""

from __future__ import annotations

from PIL import Image, ImageFilter


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply uniform Gaussian blur to an image.

    Args:
        image: PIL Image (any mode).
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        A new blurred PIL Image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))
