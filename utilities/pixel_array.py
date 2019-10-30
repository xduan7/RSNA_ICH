""" 
    File Name:          RSNA_ICH/pixel_array.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/29/19
    Python Version:     3.5.4
    File Description:   

"""
from typing import Optional
from .constants import *


def get_pixel_array(
        file_name: str,
) -> Optional[np.ndarray]:
    # Modify the file name so that it starts with "ID_" and end with ".npy"
    file_name = file_name if file_name.startswith('ID_') \
        else ('ID_' + file_name)
    file_name = file_name if file_name.endswith('.npy') \
        else (file_name + '.npy')

    # Construct file path
    pixel_array_dir = ORIGINAL_PIXEL_ARRAY_DIR
    file_path = os.path.join(pixel_array_dir, file_name)

    return np.load(file_path) if os.path.exists(file_path) else None
