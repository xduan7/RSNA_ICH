""" 
    File Name:          RSNA_ICH/dicom_processing.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/25/19
    Python Version:     3.5.4
    File Description:   
        This script will go over all the dicom images and:
        (1) fetch the dicom header from each dicom;
        (2) perform the windowing, with default windowing options or
            customized ones. Windowed images will be stored in minmax
            normalized floating point numbers, in their corresponding folder;
        (3) add the following field for not-windowed image:
            * mean_pixel_value (after minmax normalization)
            * std_pixel_value (after minmax normalization)
            * num_white_pixels (== 1.0)
            * num_black_pixels (== 0.0)
            * pixel_hist[16]
"""
import os
import glob
import click
import pickle
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import List, Union, Optional, Any, Tuple, Dict

numeric = Union[int, float]

DATA_DIR = os.environ['DATA_DIR']
RAW_DIR = os.path.join(DATA_DIR, 'RSNA/ICH_detection/raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'RSNA/ICH_detection/processed')

NUM_HIST_BINS = 16

DEFAULT_WINDOW_DICT = {
    'brain_matter': {
        'window_level': 40,
        'window_range': 80,
    },
    'blood_subdural': {
        'window_level': 75,
        'window_range': 215,
    },
    'soft_tissue': {
        'window_level': 40,
        'window_range': 375,
    },
    'bone': {
        'window_level': 600,
        'window_range': 2800,
    },
}


def window_pixel_array_dict(
        pixel_array_dict: Dict[str, np.ndarray],
        window_level: numeric,
        window_range: numeric,
        verbose: bool = True,
) -> Dict[str, np.ndarray]:

    pixel_min = window_level - (window_range / 2)
    pixel_max = window_level + (window_range / 2)

    windowed_pixel_array_dict = {}

    for _dicom_name, _pixel_array in \
            tqdm(pixel_array_dict.items()) if verbose \
            else pixel_array_dict.items():

        _windowed_pixel_array = deepcopy(_pixel_array)
        _windowed_pixel_array[_windowed_pixel_array < pixel_min] = pixel_min
        _windowed_pixel_array[_windowed_pixel_array > pixel_max] = pixel_max

        windowed_pixel_array_dict[_dicom_name] = _windowed_pixel_array

    return windowed_pixel_array_dict


def get_dicom_field(
        dicom_file: pydicom.dataset.FileDataset,
        field_name: str,
        dtype: type
) -> Any:
    _field = dicom_file.get(field_name)
    if (type(_field) == pydicom.multival.MultiValue) and \
            (dtype != list) and (dtype != str):
        return dtype(_field[0])
    else:
        return dtype(_field)


def get_dicom_header_list(
        dicom_file: pydicom.dataset.FileDataset
) -> List[Union[str, int]]:
    return [
        # Various IDs
        get_dicom_field(dicom_file, 'SOPInstanceUID', str),
        get_dicom_field(dicom_file, 'PatientID', str),
        get_dicom_field(dicom_file, 'StudyInstanceUID', str),
        get_dicom_field(dicom_file, 'SeriesInstanceUID ', str),
        # Image meta data
        get_dicom_field(dicom_file, 'ImagePositionPatient', str),
        get_dicom_field(dicom_file, 'ImageOrientationPatient', str),
        get_dicom_field(dicom_file, 'PixelSpacing', str),
        # Storage specs
        get_dicom_field(dicom_file, 'BitsStored', int),
        get_dicom_field(dicom_file, 'PixelRepresentation', bool),
    ]


def get_dicom_pixel_array(
        dicom_file: pydicom.dataset.FileDataset
) -> Tuple[np.ndarray, numeric, numeric]:

    raw_pixel_array = np.array(dicom_file.pixel_array, dtype=np.float32)

    _center = get_dicom_field(dicom_file, 'WindowCenter', np.float32)
    _width = get_dicom_field(dicom_file, 'WindowWidth', np.float32)
    _slope = get_dicom_field(dicom_file, 'RescaleSlope', np.float32)
    _intercept = get_dicom_field(dicom_file, 'RescaleIntercept', np.float32)

    # Rescale but do not window the pixel array using the window
    # configuration in the header
    # The header windows are often extremely narrow
    raw_pixel_array = raw_pixel_array * _slope + _intercept

    _half_width = _width / 2
    header_pixel_min = _center - _half_width
    header_pixel_max = _center + _half_width

    return raw_pixel_array, header_pixel_min, header_pixel_max


def unpack_dicom_files(
        dicom_paths: List[str],
        header_df_path: Optional[str] = None,
        pixel_array_dict_path: Optional[str] = None,
        verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:

    os.makedirs(os.path.dirname(header_df_path), exist_ok=True)
    os.makedirs(os.path.dirname(pixel_array_dict_path), exist_ok=True)

    header_list_list, pixel_array_dict = [], {}

    print(f'Extracting header information and image array from '
          f'{len(dicom_paths)} dicom files ... ') if verbose else None

    for _dicom_path in tqdm(dicom_paths) if verbose else dicom_paths:

        try:
            _dicom_name = os.path.basename(_dicom_path)[:-4]
            _dicom_file = pydicom.dcmread(_dicom_path)

            # Note that the fields in header list is hard coded
            header_list = [_dicom_name, ] + get_dicom_header_list(_dicom_file)

            pixel_array, header_pixel_min, header_pixel_max = \
                get_dicom_pixel_array(_dicom_file)

            num_min_pixels = (pixel_array < header_pixel_min).sum()
            num_max_pixels = (pixel_array > header_pixel_max).sum()

            pixel_hist, _ = np.histogram(
                pixel_array,
                bins=NUM_HIST_BINS,
                range=(header_pixel_min, header_pixel_max))

            pixel_hist[0] += num_min_pixels
            pixel_hist[-1] += num_max_pixels

            try:
                assert pixel_hist.sum() == 512 * 512
            except AssertionError:
                print(f'The histogram does not align for {_dicom_name} with '
                      f'min = {header_pixel_min} and max = {header_pixel_max}')
                np.save(f'{_dicom_name}_pixel_array.numpy', pixel_array)

            header_list.extend(
                [header_pixel_min, header_pixel_max,
                 num_min_pixels, num_max_pixels,
                 pixel_hist])

            header_list_list.append(header_list)
            pixel_array_dict[_dicom_name] = pixel_array

        except Exception as e:
            print(f'Cant handle {_dicom_path} for {str(e)}.') \
                if verbose else None
            continue

    # Convert the header list into dataframe
    header_df = pd.DataFrame(
        header_list_list,
        columns=[
            # Dicom name that extracted from the file name by slicing
            'dicom_name',
            # IDs stored in the dicom header
            'sop_instance_uid',
            'patient_id',
            'study_instance_uid',
            'series_instance_uid',
            # Image (pixel array) meta data
            'patient_position',
            'patient_orientation',
            'pixel_spacing',
            # Storage specs
            'pixel_storage_len',
            'pixel_storage_signed',
            # Image array statistics 
            'pixel_min_value',
            'pixel_max_value',
            'num_min_pixels',
            'num_max_pixels',
            'pixel_histogram',
        ]
    )
    header_df.set_index('dicom_name', inplace=True)

    # Save all the processed data from dicom files
    if header_df_path:
        header_df.to_pickle(header_df_path)

    if pixel_array_dict_path:
        with open(pixel_array_dict_path, 'wb') as f:
            pickle.dump(pixel_array_dict, f, pickle.HIGHEST_PROTOCOL)

    return header_df, pixel_array_dict


@click.command()
@click.option('--debug', '-d', is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def process_dicom_files(debug: bool, verbose: bool):

    # reading all dicom file paths
    trn_dcm_paths = sorted(glob.glob(
        os.path.join(RAW_DIR, 'stage_1_train_images/*.dcm')))
    tst_dcm_paths = sorted(glob.glob(
        os.path.join(RAW_DIR, 'stage_1_test_images/*.dcm')))
    print(f'Number of Dicom Image Files for Training: {len(trn_dcm_paths)}')
    print(f'Number of Dicom Image Files for Testing: {len(tst_dcm_paths)}')

    if debug:
        trn_dcm_paths = trn_dcm_paths[:500]
        tst_dcm_paths = tst_dcm_paths[:100]

    for _trn_flag, _dcm_paths in \
            zip([True, False], [trn_dcm_paths, tst_dcm_paths]):

        prefix = 'trn' if _trn_flag else 'tst'

        _header_df_name = prefix + '_dcm_header_df.pickle'
        _header_df_path = os.path.join(PROCESSED_DIR, _header_df_name)

        _pixel_array_dict_name = prefix + '_pixel_array_dict.pickle'
        _pixel_array_dict_path = os.path.join(
            PROCESSED_DIR, _pixel_array_dict_name)

        # Unpack all the dicom files
        # Save the header and all "corrected" pixels into destination
        header_df, pixel_array_dict = \
            unpack_dicom_files(_dcm_paths, _header_df_path,
                               _pixel_array_dict_path, verbose)

        if debug:
            print('Head of the header dataframe: ')
            print(header_df.head())

            import random
            sample_dicom_name, sample_dicom_pixel_array = \
                random.choice(list(pixel_array_dict.items()))

            sample_dicom_pixel_array_list = [sample_dicom_pixel_array, ]

        # Create windowed pixel arrays using default window config
        for _window_name, _window_config in DEFAULT_WINDOW_DICT.items():

            if verbose:
                print(f'Windowing pixel arrays by f{_window_name} ...')

            _window_level = _window_config['window_level']
            _window_range = _window_config['window_range']

            _windowed_pixel_array_dict = \
                window_pixel_array_dict(pixel_array_dict,
                                        _window_level,
                                        _window_range)

            _windowed_pixel_array_dict_name = \
                prefix + f'_pixel_array_dict({_window_name}).pickle'
            _windowed_pixel_array_dict_path = os.path.join(
                PROCESSED_DIR, _windowed_pixel_array_dict_name)

            # Save the windowed pixel arrays
            with open(_windowed_pixel_array_dict_path, 'wb') as f:
                pickle.dump(_windowed_pixel_array_dict, f,
                            pickle.HIGHEST_PROTOCOL)

            if debug:
                sample_dicom_pixel_array_list.append(
                    _windowed_pixel_array_dict[sample_dicom_name])

        if debug:
            import matplotlib.pyplot as plt
            plt.figure()

            for _i in range(5):

                print(f'max: {np.max(sample_dicom_pixel_array_list[_i])}')
                print(f'min: {np.min(sample_dicom_pixel_array_list[_i])}')

                plt.subplot(2, 3, _i + 1)
                plt.imshow(sample_dicom_pixel_array_list[_i], cmap='gray')

            plt.show()


if __name__ == '__main__':
    process_dicom_files()
