""" 
    File Name:          RSNA_ICH/pixel_array_cropping.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/29/19
    Python Version:     3.5.4
    File Description:   

"""
import cv2
from copy import deepcopy
from typing import Tuple, Optional

from .constants import *


BLURRING_KERNEL_SIZE = (10, 10)
BLURRING_MASK_THRESHOLD = 0.1


def mask_pixel_array(
        pixel_array: np.ndarray,
        is_scaled: bool,
) -> Union[Tuple[np.ndarray, int, int, int, int], np.ndarray]:

    # Note that openCV requires input array to be float32 for blurring, etc.
    pixel_array = pixel_array.astype(dtype=np.float32)

    # Mask threshold needs scaling to fit into the range of the pixel array
    if is_scaled:
        _scaled_mask_threshold = BLURRING_MASK_THRESHOLD
    else:
        _scaled_mask_threshold = np.interp(
            BLURRING_MASK_THRESHOLD, (0, 1),
            (np.min(pixel_array), np.max(pixel_array)))

    _blurred_pixel_array = cv2.blur(pixel_array, BLURRING_KERNEL_SIZE)
    mask = (_blurred_pixel_array > _scaled_mask_threshold)

    _num_rows, _num_cols = pixel_array.shape
    _row_mask, _col_mask = mask.any(1), mask.any(0)
    # square_mask = np.outer(_row_mask, _col_mask)

    row_start = _row_mask.argmax()
    row_end = _num_rows - _row_mask[::-1].argmax()
    col_start = _col_mask.argmax()
    col_end = _num_cols - _col_mask[::-1].argmax()

    return mask, row_start, row_end, col_start, col_end


def mask_pixel_arrays(
        pixel_arrays: np.ndarray,
        is_scaled: bool,
) -> Tuple[np.ndarray, int, int, int, int]:

    masks = []
    ranges = []

    for _pixel_array in pixel_arrays:
        _mask, _row_start, _row_end, _col_start, _col_end = \
            mask_pixel_array(_pixel_array, is_scaled)

        masks.append(_mask)
        ranges.append([_row_start, _row_end, _col_start, _col_end])

    masks = np.array(masks, dtype=bool)
    ranges = np.array(ranges)

    row_start = np.min(ranges[:, 0])
    row_end = np.max(ranges[:, 1])
    col_start = np.min(ranges[:, 2])
    col_end = np.max(ranges[:, 3])

    return masks, row_start, row_end, col_start, col_end


def crop_pixel_array(
        pixel_array: np.ndarray,
        is_scaled: bool,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
        col_start: Optional[int] = None,
        col_end: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:

    if (not row_start) or (not row_end) or (not col_start) or (not col_end):
        mask, row_start, row_end, col_start, col_end = \
            mask_pixel_array(pixel_array, is_scaled)
    else:
        mask = None
    return mask, pixel_array[row_start: row_end, col_start: col_end]


def crop_pixel_arrays(
        pixel_arrays: np.ndarray,
        is_scaled: bool = True,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
        col_start: Optional[int] = None,
        col_end: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:

    if (not row_start) or (not row_end) or (not col_start) or (not col_end):
        _masks, row_start, row_end, col_start, col_end = \
            mask_pixel_arrays(pixel_arrays, is_scaled)
        masks = _masks[:, row_start: row_end, col_start: col_end]

    else:
        masks = None

    # cropped_pixel_arrays = np.array(
    #     [_pixel_array[row_start: row_end, col_start: col_end]
    #      for _pixel_array in pixel_arrays], dtype=PIXEL_PROCESSING_DTYPE)
    cropped_pixel_arrays = \
        pixel_arrays[:, row_start: row_end, col_start: col_end]

    return masks, cropped_pixel_arrays
