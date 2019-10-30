""" 
    File Name:          RSNA_ICH/pixel_array_regularizing.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/29/19
    Python Version:     3.5.4
    File Description:   

"""
import cv2
from .constants import *


RESIZE_INTERPOLATION = cv2.INTER_CUBIC


def regularize_pixel_arrays(
        pixel_arrays: np.ndarray,
        dimension: int,
) -> np.ndarray:

    num_channel, height, width = pixel_arrays.shape
    assert num_channel == 3

    # Pad the pixel array so that its square (3, N, N)
    if height != width:
        _square_size = height if height > width else width

        _height_padding = _square_size - height
        _height_upper_padding = _height_padding // 2
        _height_lower_padding = _height_padding - _height_upper_padding

        _width_padding = _square_size - width
        _width_upper_padding = _width_padding // 2
        _width_lower_padding = _width_padding - _width_upper_padding

        _padded_pixel_array = \
            np.pad(pixel_arrays,
                   ((0, 0),
                    (_height_upper_padding, _height_lower_padding),
                    (_width_upper_padding, _width_lower_padding)))
    else:
        _padded_pixel_array = pixel_arrays

    # Resize into the desired dimension
    regularized_pixel_arrays = []
    for _channel_pixel_array in _padded_pixel_array:

        _resized_pixel_array = cv2.resize(
            _channel_pixel_array,
            dsize=(dimension, dimension),
            interpolation=cv2.INTER_CUBIC)
        regularized_pixel_arrays.append(_resized_pixel_array)

    return np.array(regularized_pixel_arrays, dtype=PIXEL_PROCESSING_DTYPE)
