""" 
    File Name:          RSNA_ICH/dicom_processing.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/28/19
    Python Version:     3.5.4
    File Description:   

"""
import glob
import pydicom
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Any, Tuple

from .constants import *


NUM_HIST_BINS = 16
DCM_INFO_HEADERS = [
    # [old name, new name, dtype (for interpretation)]
    # Various IDs
    ['SOPInstanceUID', 'sop_instance_uid', str],
    ['PatientID', 'patient_id', str],
    ['StudyInstanceUID', 'study_instance_uid', str],
    ['SeriesInstanceUID', 'series_instance_uid',  str],
    # Image meta data
    ['ImagePositionPatient', 'patient_position', str],
    ['ImageOrientationPatient', 'patient_orientation', str],
    ['PixelSpacing', 'pixel_spacing', str],
    # Storage specs
    ['BitsStored', 'pixel_storage_len', int],
    ['PixelRepresentation', 'pixel_storage_signed', bool],
    [None, 'num_rows', np.int32],
    [None, 'num_cols', np.int32],
    # Pixel array statistics (based on dicom header specified windowing)
    [None, 'pixel_min_value', np.int32],
    [None, 'pixel_max_value', np.int32],
    [None, 'num_min_pixels', np.int32],
    [None, 'num_max_pixels', np.int32],
    *[[None, f'pixel_histogram_bin_{_i_bin}', np.int32]
      for _i_bin in range(NUM_HIST_BINS)],
]


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


def get_dicom_header(
        dicom_file: pydicom.dataset.FileDataset
) -> List[Union[str, int]]:
    # return [
    #     # Various IDs
    #     get_dicom_field(dicom_file, 'SOPInstanceUID', str),
    #     get_dicom_field(dicom_file, 'PatientID', str),
    #     get_dicom_field(dicom_file, 'StudyInstanceUID', str),
    #     get_dicom_field(dicom_file, 'SeriesInstanceUID ', str),
    #     # Image meta data
    #     get_dicom_field(dicom_file, 'ImagePositionPatient', str),
    #     get_dicom_field(dicom_file, 'ImageOrientationPatient', str),
    #     get_dicom_field(dicom_file, 'PixelSpacing', str),
    #     # Storage specs
    #     get_dicom_field(dicom_file, 'BitsStored', int),
    #     get_dicom_field(dicom_file, 'PixelRepresentation', bool),
    # ]
    return [get_dicom_field(dicom_file, _dih[0], _dih[2])
            for _dih in DCM_INFO_HEADERS if _dih[0]]


def get_dicom_pixel_array(
        dicom_file: pydicom.dataset.FileDataset
) -> Tuple[np.ndarray, Numeric, Numeric]:

    # If the pixel array cannot be converted into int16 (out of range)
    if (np.max(dicom_file.pixel_array) > np.iinfo(np.int16).max) or \
            (np.min(dicom_file.pixel_array) < np.iinfo(np.int16).min):
        print(f'Data overflown for dtype {PIXEL_STORAGE_DTYPE}!')

    raw_pixel_array = np.array(dicom_file.pixel_array, dtype=PIXEL_STORAGE_DTYPE)

    _center = get_dicom_field(dicom_file, 'WindowCenter', PIXEL_STORAGE_DTYPE)
    _width = get_dicom_field(dicom_file, 'WindowWidth', PIXEL_STORAGE_DTYPE)
    _slope = get_dicom_field(dicom_file, 'RescaleSlope', PIXEL_STORAGE_DTYPE)
    _intercept = get_dicom_field(dicom_file, 'RescaleIntercept', PIXEL_STORAGE_DTYPE)

    # Rescale but do not window the pixel array using the window
    # configuration in the header
    # The header windows are often extremely narrow
    raw_pixel_array = raw_pixel_array * _slope + _intercept

    _half_width: PIXEL_STORAGE_DTYPE = _width // 2
    header_pixel_min: PIXEL_STORAGE_DTYPE = _center - _half_width
    header_pixel_max: PIXEL_STORAGE_DTYPE = _center + _half_width

    return raw_pixel_array, header_pixel_min, header_pixel_max


def unpack_dicoms(
        dicom_paths: List[str],
        header_df_path: Optional[str] = None,
        verbose: bool = True
) -> pd.DataFrame:

    os.makedirs(os.path.dirname(header_df_path), exist_ok=True)
    os.makedirs(PIXEL_ARRAY_DIR, exist_ok=True)

    headers = []

    print(f'Extracting header information and image array from '
          f'{len(dicom_paths)} dicom files ... ') if verbose else None

    for _dicom_path in tqdm(dicom_paths) if verbose else dicom_paths:

        try:
            _dicom_name = os.path.basename(_dicom_path)[:-4]
            _dicom_file = pydicom.dcmread(_dicom_path)

            # Note that the fields in header list is hard coded
            header = [_dicom_name, ] + get_dicom_header(_dicom_file)

            # The shape of pixel array is not always (512, 512)
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

            header.extend(
                [*pixel_array.shape,
                 header_pixel_min, header_pixel_max,
                 num_min_pixels, num_max_pixels,
                 *pixel_hist])

            # Append the header and save the pixel array
            headers.append(header)
            # Note that all pixel arrays are saved in the same directory
            # No matter if they are for training and test
            np.save(os.path.join(PIXEL_ARRAY_DIR, _dicom_name),
                    arr=pixel_array)

        except Exception as e:
            print(f'Cant handle {_dicom_path} for {str(e)}.') \
                if verbose else None
            continue

    # Convert the header list into dataframe
    header_df = pd.DataFrame(
        headers,
        columns=(['dicom_name', ] + [_dih[1] for _dih in DCM_INFO_HEADERS]))
    header_df.set_index('dicom_name', inplace=True)

    # Save all the processed data from dicom files
    if header_df_path:
        header_df.to_pickle(header_df_path)

    return header_df


def process_dicoms(
        verbose: bool
) -> None:
    """
    This function will go over all the dicom images and:
    (1) fetch the dicom header from each dicom;
    (2) perform the windowing, with default windowing options or
        customized ones. Windowed images will be stored in minmax
        normalized floating point numbers, in their corresponding folder;
    (3) add the following field for not-windowed image:
        * number of rows
        * number of cols
        * minimum value of pixels (after default windowing)
        * maximum value of pixels (after default windowing)
        * number of pixels with minimum value
        * number of pixels with maximum value
        * number of pixels in histogram (num_bins = NUM_HIST_BINS)

    :param verbose:
    :return:
    """

    if os.path.exists(TRN_HDR_DF_PATH) and os.path.exists(TST_HDR_DF_PATH):
        if verbose:
            print(f'Skipping dicom processing for '
                  f'both {TRN_HDR_DF_PATH} and {TST_HDR_DF_PATH} exist ...')
        return

    # reading all dicom file paths
    trn_dcm_paths = sorted(glob.glob(
        os.path.join(RAW_DIR, 'stage_1_train_images/*.dcm')))
    tst_dcm_paths = sorted(glob.glob(
        os.path.join(RAW_DIR, 'stage_1_test_images/*.dcm')))
    print(f'Number of Dicom Image Files for Training: {len(trn_dcm_paths)}')
    print(f'Number of Dicom Image Files for Testing: {len(tst_dcm_paths)}')

    for _trn_flag, _dcm_paths in \
            zip([True, False], [trn_dcm_paths, tst_dcm_paths]):

        # prefix = 'trn' if _trn_flag else 'tst'
        # _header_df_name = prefix + '_dcm_header_df.pickle'
        _header_df_name = TRN_HDR_DF_PATH if _trn_flag else TST_HDR_DF_PATH
        _header_df_path = os.path.join(PROCESSED_DIR, _header_df_name)

        if not os.path.exists(_header_df_path):
            # Unpack all the dicom files
            # Save the header and all "corrected" pixels into destination
            unpack_dicoms(_dcm_paths, _header_df_path, verbose)

    # # Old windowing code:
    # Create windowed pixel arrays using default window config
    # if window_name and window_level and window_range:
    #     window_names, window_levels, window_ranges = \
    #         [window_name, ], [window_level, ], [window_range, ]
    # else:
    #     window_names, window_levels, window_ranges = [], [], []
    #
    # window_pixel_arrays(
    #     window_names, window_levels, window_ranges, verbose)
