""" 
    File Name:          RSNA_ICH/plot_windows.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/4/19
    Python Version:     3.5.4
    File Description:   

"""
from typing import Optional

import pandas as pd

from utilities.ich_dataset import ICHDataset, WindowRanges
from utilities.constants import *
from utilities.pixel_array_windowing import DEFAULT_WINDOW_RANGES
from utilities.csv_processing import load_trn_lbl_df

error_df = pd.read_csv('./errors.txt', header=None, index_col=0, sep=' ')

#  image_id, loss_rank, loss, ground_truth, predicted_probabilities

label_to_num = {
    'any': 0,
    'epidural': 1,
    'subdural': 2,
    'subarachnoid': 3,
    'intraventricular': 4,
    'intraparenchymal': 5,
}


error_df.index.name = 'id'
error_df.columns = ['loss_rank', 'loss', 'ground_truth', ] + \
    [('predicted_' + _d) for _d in label_to_num.keys()]


trn_lbl_df = load_trn_lbl_df()
trn_hdr_df = pd.read_pickle(TRN_HDR_DF_PATH)
tst_hdr_df = pd.read_pickle(TST_HDR_DF_PATH)

trn_df = pd.concat([trn_hdr_df, trn_lbl_df],
                   axis=1,
                   join='inner')

all_hdr_df = pd.concat([trn_hdr_df, tst_hdr_df],
                       axis=1,
                       join='outer',
                       sort=True)


dset_kwargs = {
    'training': False,
    'dataframe': all_hdr_df,
    'window_ranges': DEFAULT_WINDOW_RANGES,
    'equalization': True,
    'regularize_dim': 512,
    'low_memory': True,
}

dset = ICHDataset(**dset_kwargs, transform=None)

APPIAN_WINDOW_RANGES = [
    [0, 80, True],
    [-20, 180, True],
    [-150, 230, True],
]

def plot_error_image(
        error_index: int,
        error_id: Optional[str] = None,
        window_ranges: Optional[WindowRanges] = None
):
    if error_id is None:
        error_id = error_df.iloc[error_index].name

    print(error_df.loc[error_id])

    _index = all_hdr_df.index.get_loc(error_id)
    dset.getitem(
        _index,
        demonstration=True,
        equalize_num_bins=256,
        equalize_mask_usage=True,
        equalize_adaption=False,
        window_ranges=window_ranges if window_ranges
        else DEFAULT_WINDOW_RANGES)
