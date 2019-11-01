""" 
    File Name:          RSNA_ICH/pixel_array_equalizing.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/1/19
    Python Version:     3.5.4
    File Description:   

"""
from skimage.exposure import equalize_hist, equalize_adapthist

from .constants import *
from .pixel_array_cropping import mask_pixel_array


EQUALIZE_NUM_BINS = 256
EQUALIZE_USE_MASK = True
EQUALIZE_ADAPTION = False


def equalize_pixel_arrays(
        pixel_arrays: np.ndarray,
        num_bins: int = EQUALIZE_NUM_BINS,
        mask_usage: bool = EQUALIZE_USE_MASK,
        adaption: bool = EQUALIZE_ADAPTION,
        adaptive_clip_limit: float = 0.01

):
    adjusted_pixel_array = []
    for _pixel_array in pixel_arrays:

        _mask = mask_pixel_array(_pixel_array,
                                 is_scaled=True,
                                 return_mask=True) \
            if mask_usage else None

        eq_hist_kwargs = {
            'image': _pixel_array,
            'nbins': num_bins
        }

        adjusted_pixel_array.append(
            equalize_adapthist(**eq_hist_kwargs,
                               clip_limit=adaptive_clip_limit)
            if adaption else equalize_hist(**eq_hist_kwargs,
                                           mask=_mask))

    return np.array(adjusted_pixel_array, dtype=PIXEL_PROCESSING_DTYPE)
