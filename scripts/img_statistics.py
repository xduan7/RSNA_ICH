# Go over all the 512x512 PNG files
# Histogram for pixels bin=[0, 255] (for each category)
# Histogram for the number of valid points in the pixel (for each category)
# Mean and variation of each pixel (for better windowing)

import os
import cv2
import click
import pickle
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from addict import Dict
from copy import deepcopy
from dataframe_conversion import original_csv_to_dataframe

ICH_DIR = os.path.join(os.environ['DATA_DIR'], 'RSNA/ICH_detection')
SUB_CATEGORIES = ['epidural', 'intraparenchymal', 'intraventricular',
                  'subarachnoid', 'subdural', ]
CATEGORIES = ['all', 'any', 'healthy'] + SUB_CATEGORIES
DEFAULT_STATS_PATH = os.path.join(ICH_DIR, 'img_statistics.pickle')


@click.command()
@click.option('--stats_path', '-o', type=str, default=DEFAULT_STATS_PATH)
@click.option('--rez', '-r', type=int, default=512)
@click.option('--debug', '-d', is_flag=True)
def analyze_png(stats_path: str, rez: int, debug: bool):
    trn_png_dir = os.path.join(ICH_DIR, f'png_{rez}/stage_1_train_images')
    tst_png_dir = os.path.join(ICH_DIR, f'png_{rez}/stage_1_test_images')

    trn_csv_path = os.path.join(ICH_DIR, 'stage_1_train.csv')
    trn_df_path = os.path.join(ICH_DIR, 'stage_1_train.pickle')
    tst_csv_path = os.path.join(ICH_DIR, 'stage_1_sample_submission.csv')
    tst_df_path = os.path.join(ICH_DIR, 'stage_1_sample_submission.pickle')

    trn_png_paths = glob(os.path.join(trn_png_dir, '*.png'))
    tst_png_paths = glob(os.path.join(tst_png_dir, '*.png'))

    trn_df = original_csv_to_dataframe(csv_path=trn_csv_path,
                                       png_paths=trn_png_paths,
                                       df_path=trn_df_path)
    tst_df = original_csv_to_dataframe(csv_path=tst_csv_path,
                                       png_paths=tst_png_paths,
                                       df_path=tst_df_path)

    if debug:
        pd.set_option('display.max_columns', None)
        print('Head of the training dataframe:')
        print(trn_df.head())
        print('Head of the test dataframe:')
        print(tst_df.head())
        trn_df = trn_df[: 1000]
        tst_df = tst_df[: 1000]

    category_stat_dict = Dict({
        'cnt': np.uint64(0),
        'img': np.zeros(shape=(rez, rez), dtype=np.uint64),
        'brt_hist': np.zeros(shape=(256, ), dtype=np.uint64),
        'pxl_hist': np.zeros(shape=(256, ), dtype=np.uint64)
    })

    stats = Dict({'trn': Dict(), 'tst': Dict()})

    for _trn, _df in zip([True, False], [trn_df, tst_df]):

        if _trn:
            _stats_dict = stats.trn
            for _c in CATEGORIES:
                _stats_dict[_c] = deepcopy(category_stat_dict)
            _png_dir = trn_png_dir
        else:
            _stats_dict = stats.tst
            _stats_dict.all = deepcopy(category_stat_dict)
            _png_dir = tst_png_dir

        for _i in tqdm(_df.index):

            # Tags for each one of the images
            _tags = ['all', ]
            if float(_df.loc[_i, 'any']) == 1:
                _tags.append('any')
                for _c in SUB_CATEGORIES:
                    if float(_df.loc[_i, _c]) == 1:
                        _tags.append(_c)

            elif float(_df.loc[_i, 'any']) == 0:
                _tags.append('healthy')

            _png_path = os.path.join(
                _png_dir, _df.loc[_i, 'Image'] + '.png')
            _img = np.array(
                cv2.imread(_png_path, cv2.IMREAD_GRAYSCALE),
                dtype=np.uint8)
            _value, _cnt = np.unique(_img, return_counts=True)
            _brt = np.uint8(np.sum(_img) * 2 / (rez * rez))

            for _t in _tags:
                _stats_dict[_t].cnt += np.uint64(1)
                _stats_dict[_t].img += _img
                _stats_dict[_t].brt_hist[_brt] += np.uint64(1)
                for _v, _c in zip(_value, _cnt):
                    _stats_dict[_t].pxl_hist[_v] += _c

    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    # return stats


if __name__ == '__main__':
    analyze_png()
