""" 
    File Name:          RSNA_ICH/normalize.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/30/19
    Python Version:     3.5.4
    File Description:   

"""
import click

from utilities.ich_dataset import *


@click.command()
@click.argument('num_workers', type=int)
@click.argument('batch_size', type=int)
def normalize(
        num_workers: int,
        batch_size: int,
):

    trn_lbl_df = load_trn_lbl_df()
    trn_hdr_df = pd.read_pickle(TRN_HDR_DF_PATH)
    tst_hdr_df = pd.read_pickle(TST_HDR_DF_PATH)

    trn_df = pd.concat([trn_hdr_df, trn_lbl_df], axis=1, join='inner')

    trn_outlier_mask, tst_outlier_mask = get_outlier(trn_df, tst_hdr_df)
    valid_trn_df = trn_df[~trn_outlier_mask]

    trn_dset_kwargs = {
        'training': True,
        'dataframe': valid_trn_df,
        'regularize_dim': 512,
        'low_memory': True,
    }

    channel_avgs, channel_stds, nan_sample_ids = \
        normalize_dset(trn_dset_kwargs, num_workers, batch_size)

    print(f'Sample ID containing NaNs: {nan_sample_ids}\n'
          f'AVG: {channel_avgs.tolist()}\n'
          f'STD: {channel_stds.tolist()}')


if __name__ == '__main__':
    normalize()
