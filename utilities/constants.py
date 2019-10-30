""" 
    File Name:          RSNA_ICH/constants.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/28/19
    Python Version:     3.5.4
    File Description:   
        This file contains constants that are GLOBAL across the whole project.
        Note that the constants specific for each function are not here.
"""
import os
import numpy as np
from typing import Union

DATA_DIR = os.environ['DATA_DIR']
RAW_DIR = os.path.join(DATA_DIR, 'RSNA/ICH_detection/raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'RSNA/ICH_detection/processed')
PIXEL_ARRAY_DIR = os.path.join(PROCESSED_DIR, 'pixel_array')

TRN_LBL_CSV_PATH = os.path.join(RAW_DIR, 'stage_1_train.csv')
TRN_LBL_DF_PATH = os.path.join(PROCESSED_DIR, 'trn_lbl_df.pickle')

TRN_HDR_DF_PATH = os.path.join(PROCESSED_DIR, 'trn_hdr_df.pickle')
TST_HDR_DF_PATH = os.path.join(PROCESSED_DIR, 'tst_hdr_df.pickle')

DIAGNOSIS = [
    'any',
    'epidural',
    'intraparenchymal',
    'intraventricular',
    'subarachnoid',
    'subdural'
]

Numeric = Union[int, float]

PIXEL_STORAGE_DTYPE = np.int16
PIXEL_PROCESSING_DTYPE = np.float16
