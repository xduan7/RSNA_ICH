""" 
    File Name:          RSNA_ICH/plot_all_windows.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/27/19
    Python Version:     3.5.4
    File Description:   

"""
import os
import glob
import click
import numpy as np

DATA_DIR = os.environ['DATA_DIR']
PROCESSED_DIR = os.path.join(DATA_DIR, 'RSNA/ICH_detection/processed')


@click.command()
@click.argument('file_name', type=str)
def plot_all_windows(file_name: str):

    import matplotlib.pyplot as plt

    if not file_name.endswith('.npy'):
        file_name += '.npy'

    file_paths = glob.glob(os.path.join(PROCESSED_DIR, f'**/{file_name}'))
    print(f'Found {len(file_paths)} paths: {file_paths}')

    plt.figure(figsize=(6, 6 * len(file_paths)))

    for _i, _file_path in enumerate(file_paths):

        _pixel_array = np.load(_file_path)
        ax = plt.subplot(len(file_paths), 1,  _i + 1)
        ax.imshow(_pixel_array, cmap=plt.get_cmap('gray'))
        ax.set_title(_file_path.split('/')[-2])

    plt.show()


if __name__ == '__main__':
    plot_all_windows()
