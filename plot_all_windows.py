""" 
    File Name:          RSNA_ICH/plot_all_windows.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/27/19
    Python Version:     3.5.4
    File Description:   

"""
import click
from utilities.pixel_array_windowing import *


@click.command()
@click.argument('file_name', type=str)
def plot_all_windows(
        file_name: str
) -> None:

    import matplotlib.pyplot as plt

    window_dirs = get_all_window_dirs()
    window_names = get_all_window_names(window_dirs)
    num_windows = len(window_names)
    pixel_arrays = [get_pixel_array(file_name, _window_name)
                    for _window_name in window_names]
    plt.figure(figsize=(10, 10 * num_windows))

    for _i, (_window_name, _pixel_array) in \
            enumerate(zip(window_names, pixel_arrays)):

        plt.subplot(num_windows, 1, _i + 1)
        plt.imshow(_pixel_array, cmap=plt.get_cmap('gray'))
        plt.title(_window_name)

    plt.show()


if __name__ == '__main__':
    plot_all_windows()
