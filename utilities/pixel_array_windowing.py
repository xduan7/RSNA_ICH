""" 
    File Name:          RSNA_ICH/pixel_array_windowing.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/28/19
    Python Version:     3.5.4
    File Description:   

"""
import glob
from tqdm import tqdm
from copy import deepcopy
from typing import List, Optional, Iterable, Tuple

from .constants import *

# DEFAULT_WINDOW_DICT = {
#     'brain_matter': {
#         'window_level': 40,
#         'window_range': 80,
#     },
#     'blood_subdural': {
#         'window_level': 75,
#         'window_range': 215,
#     },
#     'soft_tissue': {
#         'window_level': 40,
#         'window_range': 375,
#     },
#     'bone': {
#         'window_level': 600,
#         'window_range': 2800,
#     },
#
# }

WindowRanges = List[Tuple[int, int, bool]]  # last bit for inclusive
# DEFAULT_WINDOW_RANGES = [
#     (0, 80, True),  # Brain matter
#     (-20, 180, True),  # Blood/subdural
#     (-160, 240, True),  # Soft tissue
#     # [-800, 2000, True],  # Bone
# ]

DEFAULT_WINDOW_RANGES = [
    [0, 100, False],  # Brain matter
    [-100, 300, False],  # Blood/subduralå
    [200, 800, True],  # Soft tissue
    # [-800, 2000, True],  # Bone
]


def window_pixel_array(
        pixel_array: np.ndarray,
        window_ranges: WindowRanges,
        scaling: bool = True,
) -> np.ndarray:
    windowed_pixel_arrays = []

    for _window_ranges in window_ranges:

        _pixel_min = _window_ranges[0]
        _pixel_max = _window_ranges[1]
        _inclusive = _window_ranges[2]

        _windowed_pixel_array = \
            np.array(pixel_array, dtype=PIXEL_PROCESSING_DTYPE)
        _windowed_pixel_array[_windowed_pixel_array < _pixel_min] = _pixel_min
        _windowed_pixel_array[_windowed_pixel_array > _pixel_max] = \
            _pixel_max if _inclusive else _pixel_min

        if scaling:
            _windowed_pixel_array = np.interp(
                _windowed_pixel_array, (_pixel_min, _pixel_max), (0., 1.))

        windowed_pixel_arrays.append(_windowed_pixel_array)

    return np.array(windowed_pixel_arrays, dtype=PIXEL_PROCESSING_DTYPE)


# def window_pixel_array(
#         original_pixel_array: np.ndarray,
#         pixel_min: PIXEL_STORAGE_DTYPE,
#         pixel_max: PIXEL_STORAGE_DTYPE,
#         windowed_pixel_array_path: str,
# ) -> None:
#     _windowed_pixel_array = deepcopy(original_pixel_array)
#     _windowed_pixel_array[_windowed_pixel_array < pixel_min] = pixel_min
#     _windowed_pixel_array[_windowed_pixel_array > pixel_max] = pixel_max
#     np.save(windowed_pixel_array_path, arr=_windowed_pixel_array)
#
#
# def window_pixel_arrays(
#         window_names: List[str],
#         window_levels: List[Numeric],
#         window_ranges: List[Numeric],
#         verbose: bool = True,
# ) -> None:
#     assert len(window_names) == len(window_levels) == len(window_ranges)
#
#     for _window_name, _window_config in DEFAULT_WINDOW_DICT.items():
#         window_names.append(_window_name)
#         window_levels.append(_window_config['window_level'])
#         window_ranges.append(_window_config['window_range'])
#
#     # Check for necessity
#     pixel_array_paths = sorted(glob.glob(
#         os.path.join(ORIGINAL_PIXEL_ARRAY_DIR, '*')))
#     num_pixel_arrays = len(pixel_array_paths)
#     _window_names, _window_levels, _window_ranges = [], [], []
#     windowed_pixel_array_dirs = []
#     for _i in range(len(window_names)):
#
#         _windowed_pixel_array_dir = os.path.join(
#             PROCESSED_DIR, f'{window_names[_i]}_pixel_array')
#         os.makedirs(_windowed_pixel_array_dir, exist_ok=True)
#         _num_windowed_pixel_arrays = len(glob.glob(os.path.join(
#             _windowed_pixel_array_dir, '*')))
#
#         if num_pixel_arrays != _num_windowed_pixel_arrays:
#             _window_names.append(window_names[_i])
#             _window_levels.append(window_levels[_i])
#             _window_ranges.append(window_ranges[_i])
#             windowed_pixel_array_dirs.append(_windowed_pixel_array_dir)
#
#     window_names, window_levels, window_ranges = \
#         _window_names, _window_levels, _window_ranges
#
#     if verbose:
#         if len(window_names) != 0:
#             print(f'Perform {len(window_names)} windowing ({window_names}) '
#                   f'on {len(pixel_array_paths)} pixel arrays ...')
#         else:
#             print('No windows for pixel arrays ...')
#
#     window_levels = np.array(window_levels, dtype=PIXEL_STORAGE_DTYPE)
#     window_ranges = np.array(window_ranges, dtype=PIXEL_STORAGE_DTYPE)
#     pixel_mins = window_levels - (window_ranges / 2)
#     pixel_maxs = window_levels + (window_ranges / 2)
#
#     for _pixel_array_path in \
#             tqdm(pixel_array_paths) if verbose else pixel_array_paths:
#
#         _pixel_array = np.load(_pixel_array_path)
#         _pixel_array_name = os.path.basename(_pixel_array_path)
#
#         for _i in range(len(window_names)):
#             _windowed_pixel_array_path = os.path.join(
#                 windowed_pixel_array_dirs[_i], f'{_pixel_array_name}')
#             window_pixel_array(_pixel_array, pixel_mins[_i], pixel_maxs[_i],
#                                _windowed_pixel_array_path)
#
#             # _windowed_pixel_array = deepcopy(_pixel_array)
#             # _windowed_pixel_array[_windowed_pixel_array < pixel_mins[_i]] = \
#             #     pixel_mins[_i]
#             # _windowed_pixel_array[_windowed_pixel_array > pixel_maxs[_i]] = \
#             #     pixel_maxs[_i]
#             # np.save(_windowed_pixel_array_path,  arr=_windowed_pixel_array)
#
#
# def get_all_window_dirs(
# ) -> List[str]:
#     return glob.glob(os.path.join(PROCESSED_DIR, '*_pixel_array*'))
#
#
# def get_all_window_names(
#         window_dirs: Optional[List[str]] = None
# ) -> List[str]:
#     window_dirs = get_all_window_dirs() if not window_dirs else window_dirs
#     window_dir_names = [os.path.basename(_d) for _d in window_dirs]
#     return [_d[: len(_d) - 12] for _d in window_dir_names]
#
#
# def get_pixel_array(
#         file_name: str,
#         window_name: Optional[str] = None,
# ) -> Optional[np.ndarray]:
#     # Modify the file name so that it starts with "ID_" and end with ".npy"
#     file_name = file_name if file_name.startswith('ID_') \
#         else ('ID_' + file_name)
#     file_name = file_name if file_name.endswith('.npy') \
#         else (file_name + '.npy')
#
#     # Construct file path
#     pixel_array_dir = ORIGINAL_PIXEL_ARRAY_DIR if (not window_name) else \
#         os.path.join(PROCESSED_DIR, f'{window_name}_pixel_array')
#     file_path = os.path.join(pixel_array_dir, file_name)
#
#     return np.load(file_path) if os.path.exists(file_path) else None
