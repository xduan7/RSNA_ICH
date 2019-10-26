""" 
    File Name:          RSNA_ICH/dicom_to_png.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/19/19
    Python Version:     3.5.4
    File Description:   

"""

import os
import click
import pydicom
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image


ICH_DIR = os.path.join(os.environ['DATA_DIR'], 'RSNA/ICH_detection')
DICOM_DIR = os.path.join(ICH_DIR, 'dicom')


@click.command()
@click.argument('resolution', type=int)
def dicom_to_png(resolution):

    # reading all dcm files into train and text
    trn_dcm_paths = sorted(glob(
        os.path.join(DICOM_DIR, 'stage_1_train_images/*.dcm')))
    tst_dcm_paths = sorted(glob(
        os.path.join(DICOM_DIR, 'stage_1_test_images/*.dcm')))
    print(f'Number of Dicom Image Files for Training: {len(trn_dcm_paths)}')
    print(f'Number of Dicom Image Files for Testing: {len(tst_dcm_paths)}')
    print(f'Target Image (PNG) Resolution: ({resolution}, {resolution})')

    # trn_df = pd.read_csv(os.path.join(ICH_DIR, 'stage_1_train.csv'))

    os.makedirs(os.path.join(
        ICH_DIR, f'png_{resolution}', 'stage_1_train_images'), exist_ok=True)
    os.makedirs(os.path.join(
        ICH_DIR, f'png_{resolution}', 'stage_1_test_images'), exist_ok=True)

    _dcm_paths = trn_dcm_paths + tst_dcm_paths

    for _dcm_path in tqdm(_dcm_paths):

        _png_path = _dcm_path.replace('/dicom/', f'/png_{resolution}/')\
            .replace('.dcm', '.png')

        if os.path.exists(_png_path):
            continue

        try:
            _dcm = pydicom.dcmread(_dcm_path)

            # Dicom image and other info
            _dcm_img = _dcm.pixel_array

            def get_dcm_filed(__dcm: pydicom.FileDataset, __field_name: str):
                __field = __dcm.get(__field_name)
                if type(__field) == pydicom.multival.MultiValue:
                    return int(__field[0])
                else:
                    return int(__field)

            _dcm_center = get_dcm_filed(_dcm, 'WindowCenter')
            _dcm_width = get_dcm_filed(_dcm, 'WindowWidth')
            _dcm_intercept = get_dcm_filed(_dcm, 'RescaleIntercept')
            _dcm_slope = get_dcm_filed(_dcm, 'RescaleSlope')
            _dcm_info = [_dcm_center, _dcm_width, _dcm_intercept, _dcm_slope]

            # Scaled dicom image
            _dcm_img = (_dcm_img * _dcm_slope + _dcm_intercept)
            _dcm_img_min = _dcm_center - _dcm_width // 2
            _dcm_img_max = _dcm_center + _dcm_width // 2
            _dcm_img[_dcm_img < _dcm_img_min] = _dcm_img_min
            _dcm_img[_dcm_img > _dcm_img_max] = _dcm_img_max

            # Greyscale, resize and save
            _png_arr = np.uint8(255. * (
                    (_dcm_img - _dcm_img_min) / (_dcm_img_max - _dcm_img_min)))

            _png_img = Image.fromarray(_png_arr, mode='L')
            _png_img = _png_img.resize((resolution, resolution))
            _png_img.save(_png_path)

            # Pause for testing
            # input("Press Enter to continue...")

        except Exception as e:
            print(f'Could not handle {_dcm_path}. \n'
                  f'Error message: {str(e)}')
            continue


if __name__ == '__main__':
    dicom_to_png()
