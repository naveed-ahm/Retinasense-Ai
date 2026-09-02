"""Lightweight gate for retinal fundus photographs.

This deliberately uses only image characteristics and does not load or alter
the disease-prediction model.
"""
import cv2
import numpy as np


INVALID_FUNDUS_MESSAGE = (
    "Invalid Image\n\n"
    "Please upload a valid retinal fundus image for disease screening."
)


def is_fundus_image(image_rgb: np.ndarray) -> bool:
    """Return whether an RGB image has the colour profile of a fundus field.

    Fundus photographs are dominated by a saturated red/orange retinal field
    captured within a circular or elliptical camera field.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return False
    height, width = image_rgb.shape[:2]
    if min(height, width) < 128:
        return False

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    retinal_tone = (
        (hue <= 35)
        & (saturation >= 45)
        & (value >= 25)
        & (image_rgb[:, :, 0] > image_rgb[:, :, 1] * 1.05)
        & (image_rgb[:, :, 1] >= image_rgb[:, :, 2] * 0.70)
    )

    # Fundus cameras produce a bounded circular/elliptical field against a
    # black background. Colour alone would incorrectly admit a red/orange
    # sunset, so verify this capture geometry before allowing prediction.
    non_black = (value >= 15).astype(np.uint8)
    contours, _ = cv2.findContours(non_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    field = max(contours, key=cv2.contourArea)
    field_area = cv2.contourArea(field)
    x, y, field_width, field_height = cv2.boundingRect(field)
    bounding_area = field_width * field_height
    if bounding_area == 0:
        return False
    field_fill_ratio = field_area / bounding_area
    if not 0.55 <= field_fill_ratio <= 0.90:
        return False

    field_mask = np.zeros(non_black.shape, dtype=np.uint8)
    cv2.drawContours(field_mask, [field], -1, 1, thickness=cv2.FILLED)
    field_pixels = int(np.count_nonzero(field_mask))
    if field_pixels == 0:
        return False
    retinal_tone_ratio = float(np.count_nonzero(retinal_tone & field_mask.astype(bool))) / field_pixels
    return retinal_tone_ratio >= 0.55
