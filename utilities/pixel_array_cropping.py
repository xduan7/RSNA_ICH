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
) -> Tuple[int, int, int, int]:

    # Mask threshold needs scaling to fit into the range of the pixel array
    if is_scaled:
        _scaled_mask_threshold = BLURRING_MASK_THRESHOLD
    else:
        _scaled_mask_threshold = np.interp(
            BLURRING_MASK_THRESHOLD, (0, 1),
            (np.min(pixel_array), np.max(pixel_array)))

    _blurred_pixel_array = cv2.blur(pixel_array, BLURRING_KERNEL_SIZE)
    _mask = (_blurred_pixel_array > _scaled_mask_threshold)

    _num_rows, _num_cols = pixel_array.shape
    _row_mask, _col_mask = _mask.any(1), _mask.any(0)
    # _mask = np.outer(_row_mask, _col_mask)

    row_start = _row_mask.argmax()
    row_end = _num_rows - _row_mask[::-1].argmax()
    col_start = _col_mask.argmax()
    col_end = _num_cols - _col_mask[::-1].argmax()

    return row_start, row_end, col_start, col_end


def mask_pixel_arrays(
        pixel_arrays: np.ndarray,
        is_scaled: bool,
) -> Tuple[int, int, int, int]:

    _mask_indices = np.array(
        [list(mask_pixel_array(_pixel_array, is_scaled))
         for _pixel_array in pixel_arrays])

    row_start = np.min(_mask_indices[:, 0])
    row_end = np.max(_mask_indices[:, 1])
    col_start = np.min(_mask_indices[:, 2])
    col_end = np.max(_mask_indices[:, 3])

    return row_start, row_end, col_start, col_end


def crop_pixel_array(
        pixel_array: np.ndarray,
        is_scaled: bool,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
        col_start: Optional[int] = None,
        col_end: Optional[int] = None,
) -> np.ndarray:

    if (not row_start) or (not row_end) or (not col_start) or (not col_end):
        row_start, row_end, col_start, col_end = \
            mask_pixel_array(pixel_array, is_scaled)
    return deepcopy(pixel_array[row_start: row_end, col_start: col_end])


def crop_pixel_arrays(
        pixel_arrays: np.ndarray,
        is_scaled: bool = True,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
        col_start: Optional[int] = None,
        col_end: Optional[int] = None,
) -> np.ndarray:

    if (not row_start) or (not row_end) or (not col_start) or (not col_end):
        row_start, row_end, col_start, col_end = \
            mask_pixel_arrays(pixel_arrays, is_scaled)

    return np.array(
        [deepcopy(_pixel_array[row_start: row_end, col_start: col_end])
         for _pixel_array in pixel_arrays], dtype=PIXEL_PROCESSING_DTYPE)
