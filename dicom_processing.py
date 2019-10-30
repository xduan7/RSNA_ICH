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
            * ~~mean_pixel_value (after minmax normalization)~~
            * ~~std_pixel_value (after minmax normalization)~~
            * num_white_pixels (== 1.0)
            * num_black_pixels (== 0.0)
            * pixel_hist[16]
"""
import glob
import click
from typing import Optional

from utilities.constants import *
from utilities.dicom_parsing import unpack_dicom_files
# from utilities.pixel_array_windowing import window_pixel_arrays


@click.command()
@click.option('--window_name', '-n', type=str, default=None)
@click.option('--window_level', '-l', type=int, default=None)
@click.option('--window_range', '-r', type=int, default=None)
@click.option('--debug', '-d', is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def process_dicom_files(
        window_name: Optional[str],
        window_level: Optional[int],
        window_range: Optional[int],
        debug: bool,
        verbose: bool
) -> None:

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

        # prefix = 'trn' if _trn_flag else 'tst'
        # _header_df_name = prefix + '_dcm_header_df.pickle'
        _header_df_name = TRN_HDR_DF_PATH if _trn_flag else TST_HDR_DF_PATH
        _header_df_path = os.path.join(PROCESSED_DIR, _header_df_name)

        if not os.path.exists(_header_df_path):
            # Unpack all the dicom files
            # Save the header and all "corrected" pixels into destination
            unpack_dicom_files(_dcm_paths, _header_df_path, verbose)

    # Create windowed pixel arrays using default window config
    # if window_name and window_level and window_range:
    #     window_names, window_levels, window_ranges = \
    #         [window_name, ], [window_level, ], [window_range, ]
    # else:
    #     window_names, window_levels, window_ranges = [], [], []
    #
    # window_pixel_arrays(
    #     window_names, window_levels, window_ranges, verbose)


if __name__ == '__main__':
    process_dicom_files()
