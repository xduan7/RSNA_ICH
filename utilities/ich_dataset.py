""" 
    File Name:          RSNA_ICH/ich_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/21/19
    Python Version:     3.5.4
    File Description:   

"""
import torch
import pandas as pd
import albumentations
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from .pixel_array_equalizing import *
from .pixel_array_cropping import *
from .pixel_array_regularizing import *
from .pixel_array_windowing import *
from .constants import *
from .csv_processing import load_trn_lbl_df
from .pixel_array import get_pixel_array

MIN_PIXELS_THOLD = 0.84
NUM_CV_FOLDS = 5
STRATIFIED_COLS = DIAGNOSIS


def get_outlier(
        trn_df: pd.DataFrame,
        tst_hdr_df: pd.DataFrame,
        min_pixels_thold: float = MIN_PIXELS_THOLD,
):
    # Outlier masks: marked true if the sample identified as outlier
    trn_outlier_mask = trn_df['num_min_pixels'] > \
                       (min_pixels_thold * trn_df['num_rows'] * trn_df[
                           'num_cols'])
    tst_outlier_mask = tst_hdr_df['num_min_pixels'] > \
                       (min_pixels_thold * tst_hdr_df['num_rows'] * tst_hdr_df[
                           'num_cols'])

    # Print out the number of training samples, outliers and the number of
    # patients in the outliers
    _num_trn = len(trn_df)
    _num_trn_outliers = trn_outlier_mask.sum()
    _num_patients_as_outliers = trn_df[trn_outlier_mask]['any'].sum()

    print(f'{_num_trn_outliers / _num_trn * 100.:.2f}% '
          f'({_num_trn_outliers}/{_num_trn}) '
          f'training samples identified as outliers, '
          f'{_num_patients_as_outliers / _num_trn_outliers * 100.:.2f}% '
          f'({_num_patients_as_outliers}) '
          f'of which are labeled as patients.')

    return trn_outlier_mask, tst_outlier_mask


def get_n_fold_trn_vld_ids(

        stratified_cols: List[str] = STRATIFIED_COLS,
        min_pixels_thold: float = MIN_PIXELS_THOLD,
):
    # Load the training dataframe (label + header)
    trn_lbl_df = load_trn_lbl_df()
    trn_hdr_df = pd.read_pickle(TRN_HDR_DF_PATH)
    tst_hdr_df = pd.read_pickle(TST_HDR_DF_PATH)

    # The new trn_df has all the header and label information
    trn_df = pd.concat([trn_hdr_df, trn_lbl_df], axis=1, join='inner')

    trn_outlier_mask, tst_outlier_mask = \
        get_outlier(trn_df, tst_hdr_df, min_pixels_thold)

    valid_trn_df = trn_df[~trn_outlier_mask]
    valid_trn_ids = valid_trn_df.index

    stratified_labels = np.zeros(shape=(len(valid_trn_df),), dtype=int)
    for _i, _col in enumerate(stratified_cols):
        _le = LabelEncoder()
        _col_values = valid_trn_df[_col].values
        _stratified_label = _le.fit_transform(_col_values)

        stratified_labels = \
            stratified_labels * len(_le.classes_) + _stratified_label

    # Do not shuffle for a fixed state
    splitter = StratifiedKFold(n_splits=NUM_CV_FOLDS)
    n_fold_trn_vld_ids = \
        [[valid_trn_ids[_trn_ids].to_list(),
          valid_trn_ids[_vld_ids].to_list()]
         for _trn_ids, _vld_ids in
         splitter.split(valid_trn_ids, stratified_labels)]

    return n_fold_trn_vld_ids


class ICHDataset(Dataset):

    def __init__(
            self,
            training: bool,
            dataframe: pd.DataFrame,
            window_ranges: WindowRanges = DEFAULT_WINDOW_RANGES,

            equalization: bool = True,
            equalize_num_bins: int = EQUALIZE_NUM_BINS,
            equalize_mask_usage: bool = EQUALIZE_USE_MASK,
            equalize_adaption: bool = EQUALIZE_ADAPTION,

            regularize_dim: Optional[int] = None,

            transform: Optional[albumentations.Compose] = None,
            low_memory: bool = True,
    ):

        self.__training = training
        self.__df = dataframe
        self.__ids = list(dataframe.index)

        self.__window_ranges = window_ranges
        self.__regularize_dim = regularize_dim

        self.__equalization = equalization
        self.__equalize_num_bins = equalize_num_bins
        self.__equalize_mask_usage = equalize_mask_usage
        self.__equalize_adaption = equalize_adaption

        self.__transform = transform
        self.__low_memory = low_memory

        self.__len = len(self.__ids)

        self.__pixel_array_dict = {}
        self.__lbl_dict = {}

    def __len__(self):
        return self.__len

    def getitem(
            self,
            index: int,
            demonstration: bool,
            equalize_num_bins: Optional[int] = None,
            equalize_mask_usage: Optional[bool] = None,
            equalize_adaption: Optional[bool] = None,
            window_ranges: Optional[WindowRanges] = None,
    ):
        # ID (SOP Instance UID) of the sample
        _id: str = self.__ids[index]

        # Get original pixel array
        if self.__low_memory and (not demonstration) and \
                index in self.__pixel_array_dict:
            _pixel_array = self.__pixel_array_dict[_id]
        else:
            _original_pixel_array = get_pixel_array(_id).astype(np.float32)

            # Pixel array processing (windowing, cropping, regularization)
            _window_ranges = window_ranges if demonstration \
                else self.__window_ranges

            _windowed_pixel_arrays = \
                window_pixel_array(_original_pixel_array,
                                   window_ranges=_window_ranges,
                                   scaling=True)

            _masks, _cropped_pixel_arrays = \
                crop_pixel_arrays(_windowed_pixel_arrays,
                                  is_scaled=True)

            _equalize_num_bins = equalize_num_bins if demonstration \
                else self.__equalize_num_bins
            _equalize_mask_usage = equalize_mask_usage if demonstration \
                else self.__equalize_mask_usage
            _equalize_adaption = equalize_adaption if demonstration \
                else self.__equalize_adaption
            if self.__equalization:
                _equalized_pixel_array = \
                    equalize_pixel_arrays(_cropped_pixel_arrays,
                                          masks=_masks if
                                          _equalize_mask_usage else None,
                                          num_bins=_equalize_num_bins,
                                          adaption=_equalize_adaption)
            else:
                _equalized_pixel_array = _cropped_pixel_arrays

            _regularized_pixel_array = \
                regularize_pixel_arrays(_equalized_pixel_array,
                                        dimension=self.__regularize_dim)

            if demonstration:

                print(f'Index = {index}')
                print(self.__df.loc[_id])

                import matplotlib.pyplot as plt
                plt.figure(figsize=(20, 24))

                plt.subplot(5, 4, 1)
                plt.imshow(_original_pixel_array, cmap=plt.get_cmap('gray'))
                plt.title('original pixel array')

                plt.subplot(5, 4, 5)
                plt.imshow(np.transpose(_windowed_pixel_arrays, (1, 2, 0)))
                plt.title(f'windowed (RGB)')
                for _i in range(len(_windowed_pixel_arrays)):
                    plt.subplot(5, 4, 6 + _i)
                    plt.imshow(_windowed_pixel_arrays[_i],
                               cmap=plt.get_cmap('gray'))
                    plt.title(f'windowed {str(window_ranges[_i])}')

                plt.subplot(5, 4, 9)
                plt.imshow(np.transpose(_cropped_pixel_arrays, (1, 2, 0)))
                plt.title(f'cropped (RGB)')
                for _i in range(len(_cropped_pixel_arrays)):
                    plt.subplot(5, 4, 10 + _i)
                    plt.imshow(_cropped_pixel_arrays[_i],
                               cmap=plt.get_cmap('gray'))
                    plt.title(
                        f'cropped {str(_cropped_pixel_arrays[_i].shape)}')

                if self.__equalization:
                    plt.subplot(5, 4, 13)
                    plt.imshow(np.transpose(_equalized_pixel_array, (1, 2, 0)))
                    plt.title(f'equalized (RGB)')
                    for _i in range(len(_equalized_pixel_array)):
                        plt.subplot(5, 4, 14 + _i)
                        plt.imshow(_equalized_pixel_array[_i],
                                   cmap=plt.get_cmap('gray'))
                        plt.title(f'equalized '
                                  f'mask={_equalize_mask_usage}, '
                                  f'adap={_equalize_adaption}')

                plt.subplot(5, 4, 17)
                plt.imshow(np.transpose(_regularized_pixel_array, (1, 2, 0)))
                plt.title(f'padded and resized (RGB)')
                for _i in range(len(_regularized_pixel_array)):
                    plt.subplot(5, 4, 18 + _i)
                    plt.imshow(_regularized_pixel_array[_i],
                               cmap=plt.get_cmap('gray'))
                    plt.title(f'padded and resized '
                              f'{str(_regularized_pixel_array[_i].shape)}')

                plt.show()

            _pixel_array = _regularized_pixel_array
            if not self.__low_memory:
                self.__pixel_array_dict[_id] = _pixel_array

        _image = np.moveaxis(_pixel_array, 0, -1)

        # Image augmentation and transformation
        if self.__transform:
            _image = self.__transform(image=_image)['image']

        # Get the label if the dataset is marked for training
        if self.__training:
            if index in self.__lbl_dict:
                _labels = self.__lbl_dict[index]
            else:
                _labels = torch.tensor(self.__df.loc[_id, DIAGNOSIS])
                self.__lbl_dict[index] = _labels
            return {'image': _image, 'labels': _labels, 'id': _id}
        else:
            return {'image': _image, 'id': _id}

    def __getitem__(
            self,
            index: int,
    ):
        return self.getitem(index, demonstration=False)


def normalize_dset(
        trn_dset_kwargs: dict,
        num_workers: int = -1,
        batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:

    _trn_dset_kwargs = deepcopy(trn_dset_kwargs)
    _trn_dset_kwargs['training'] = False
    _trn_dset_kwargs['transform'] = None
    _trn_dset_kwargs['low_memory'] = True

    _trn_dset = ICHDataset(**_trn_dset_kwargs)

    import multiprocessing
    num_workers = multiprocessing.cpu_count() if num_workers <= 0 \
        else max(num_workers, multiprocessing.cpu_count())

    _trn_dldr = DataLoader(_trn_dset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=False,
                           timeout=1000)

    channel_avgs = torch.zeros([3, ])
    channel_stds = torch.zeros([3, ])
    num_samples = 0
    nan_sample_ids = []
    for _batch in tqdm(_trn_dldr):

        _imgs = _batch['image']
        _num_samples = _imgs.size(0)
        _imgs = _imgs.view(_num_samples, _imgs.size(1), -1)

        _nan_masks = torch.isnan(_imgs).any(-1).any(-1)
        if _nan_masks.any():
            _nan_ids = np.array(_batch['id'])[_nan_masks].tolist()
            nan_sample_ids.extend(_nan_ids)
            print(f'Found IDs with NaN values: {_nan_ids}')

        channel_avgs += _imgs[~_nan_masks].mean(2).sum(0)
        channel_stds += _imgs[~_nan_masks].std(2).sum(0)
        num_samples += (_num_samples - _nan_masks.sum())

    channel_avgs /= num_samples
    channel_stds /= num_samples

    print(f'{num_samples}/{len(_trn_dset)}\n'
          f'AVG: {channel_avgs.tolist()}\n'
          f'STD: {channel_stds.tolist()}')

    return channel_avgs, channel_stds, nan_sample_ids
