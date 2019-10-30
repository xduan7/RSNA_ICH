""" 
    File Name:          RSNA_ICH/csv_processing.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/21/19
    Python Version:     3.5.4
    File Description:   

"""
import pandas as pd
from typing import Optional

from .constants import *


def load_trn_lbl_df(
        trn_lbl_df_path: str = TRN_LBL_DF_PATH,
) -> pd.DataFrame:

    if trn_lbl_df_path and os.path.exists(trn_lbl_df_path):
        return pd.read_pickle(trn_lbl_df_path)
    else:
        _tmp_df = pd.read_csv(TRN_LBL_CSV_PATH)

        _tmp_df[['ID', 'Image', 'Diagnosis']] = \
            _tmp_df['ID'].str.split('_', expand=True)

        _tmp_df = _tmp_df[['Image', 'Diagnosis', 'Label']]

        _tmp_df = _tmp_df.drop_duplicates()
        _tmp_df = _tmp_df.pivot(index='Image',
                                columns='Diagnosis',
                                values='Label')
        _tmp_df.index.names = ['ID']
        _tmp_df.index = 'ID_' + _tmp_df.index

        if trn_lbl_df_path:
            _tmp_df.to_pickle(trn_lbl_df_path)
        return _tmp_df


def tst_lbl_df_to_submission_csv(
        tst_lbl_df: pd.DataFrame,
        submission_csv_path: str,
) -> pd.DataFrame:

    _tmp_df = tst_lbl_df.reset_index(inplace=False)
    _tmp_df = pd.melt(_tmp_df,
                      id_vars='ID',
                      value_vars=DIAGNOSIS,
                      var_name='Diagnosis',
                      value_name='Label')

    _tmp_df['ID'] = \
        _tmp_df['ID'].map(str) + '_' + _tmp_df['Diagnosis'].map(str)
    _tmp_df = _tmp_df[['ID', 'Label']]

    _tmp_df.to_csv(submission_csv_path, index=False)

    return _tmp_df


# if __name__ == '__main__':
#
#     # Test and see if the CSV -> Dataframe -> CSV is consistent
#     original_df = pd.read_csv(TRN_LBL_CSV_PATH)
#     intermediate_df = load_trn_lbl_df(None)
#     submission_df = tst_lbl_df_to_submission_csv(intermediate_df, None)
#
#     original_df = original_df.drop_duplicates()
#
#     # Reset the index to "ID" and sort them for dataframe comparison
#     original_df.set_index('ID', inplace=True)
#     submission_df.set_index('ID', inplace=True)
#
#     original_df.sort_index(inplace=True)
#     submission_df.sort_index(inplace=True)
#
#     if (original_df == submission_df).all()['Label']:
#         print('CSV file loaded, parsed, and restored successfully!')
