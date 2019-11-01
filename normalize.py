""" 
    File Name:          RSNA_ICH/normalize.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/30/19
    Python Version:     3.5.4
    File Description:   

"""
import click
from torch.utils.data import DataLoader

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

    trn_dset = ICHDataset(training=True,
                          dataframe=valid_trn_df,
                          regularize_dim=512,
                          low_memory=True)

    trn_dldr = DataLoader(trn_dset,
                          batch_size=32,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=False,
                          timeout=0)

    # # Visualization
    # for _i in range(9):
    #     trn_dset.getitem(
    #         index=_i,
    #         demonstration=True,
    #         equalize_num_bins=256,
    #         equalize_mask_usage=True,
    #         equalize_adaption=False,
    #         window_ranges=[(-20, 180, False),
    #                        (-160, 240, False),
    #                        (160, 600, True)])

    channel_avgs = torch.zeros([3, ])
    channel_stds = torch.zeros([3, ])
    num_samples = 0
    for _batch in tqdm(trn_dldr):

        _imgs = _batch['image']
        _num_samples = _imgs.size(0)
        _imgs = _imgs.view(_num_samples, _imgs.size(1), -1)
        channel_avgs += _imgs.mean(2).sum(0)
        channel_stds += _imgs.std(2).sum(0)
        num_samples += _num_samples

        break

    channel_avgs /= num_samples
    channel_stds /= num_samples

    print(f'{num_samples}/{len(trn_dset)}\n'
          f'AVG: {channel_avgs.tolist()}\n'
          f'STD: {channel_stds.tolist()}')


if __name__ == '__main__':
    normalize()