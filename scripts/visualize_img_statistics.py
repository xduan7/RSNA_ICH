""" 
    File Name:          RSNA_ICH/visualize_img_statistics.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               10/23/19
    Python Version:     3.5.4
    File Description:   

"""
import os
import click
import pickle
from addict import Dict
from copy import deepcopy
import plotly.graph_objects as go
from plotly import tools
from plotly.subplots import make_subplots

ICH_DIR = os.path.join(os.environ['DATA_DIR'], 'RSNA/ICH_detection')
SUB_CATEGORIES = ['epidural', 'intraparenchymal', 'intraventricular',
                  'subarachnoid', 'subdural', ]
CATEGORIES = ['all', 'healthy', 'any', ] + SUB_CATEGORIES
DEFAULT_STATS_PATH = os.path.join(ICH_DIR, 'img_statistics.pickle')


LABELS = ['test', 'training', 'healthy', 'any',
          'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid',
          'subdural', ]


HEATMAP_COLORSCALE = 'Viridis'


def sunburst_plot(_stats: Dict, show=True, save=True):

    data_comp = go.Figure(go.Sunburst(
        labels=LABELS,
        parents=[
            'RSNA_ICH',
            'RSNA_ICH',
            'training',
            'training',
            'any',
            'any',
            'any',
            'any',
            'any',
        ],
        values=[_stats[_l].cnt for _l in LABELS],
        branchvalues='remainder',
    ))
    print([_stats[_l].cnt for _l in LABELS])
    data_comp.update_layout(margin=dict(t=0, l=0, r=0, b=0),
                            height=1080, width=1440)

    if show:
        data_comp.show()

    if save:
        data_comp.write_image(os.path.join(
            ICH_DIR, 'data_composition_sunburst.png'))


def avg_img(_stats: Dict, show=True, save=False):

    _avg_imgs = {_l: go.Heatmap(z=(_stats[_l].img / _stats[_l].cnt),
                                colorscale=HEATMAP_COLORSCALE,
                                name=_l)
                 for _l in LABELS}

    _img = make_subplots(
        rows=7, cols=9,
        specs=[
            [{'rowspan': 4, 'colspan': 4}, None, None, None,
             {'rowspan': 4, 'colspan': 4}, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, {'rowspan': 2, 'colspan': 2}, None,
             {'rowspan': 2, 'colspan': 2}, None, {}],
            [None, None, None, None, None, None, None, None, {}],
            [None, None, None, None, None, None, {}, {}, {}]],
        subplot_titles=LABELS)

    _img.append_trace(_avg_imgs['test'], 1, 1)
    _img.append_trace(_avg_imgs['training'], 1, 5)
    _img.append_trace(_avg_imgs['healthy'], 5, 5)
    _img.append_trace(_avg_imgs['any'], 5, 7)

    _img.append_trace(_avg_imgs['epidural'], 7, 7)
    _img.append_trace(_avg_imgs['intraparenchymal'], 7, 8)
    _img.append_trace(_avg_imgs['intraventricular'], 7, 9)
    _img.append_trace(_avg_imgs['subarachnoid'], 6, 9)
    _img.append_trace(_avg_imgs['subdural'], 5, 9)

    _img.update_layout(height=1080, width=1440,
                       title_text='Averaged Pixels',
                       showlegend=False)

    if show:
        _img.show()

    if save:
        _img.write_image(os.path.join(
            ICH_DIR, 'average_image.png'))


def histogram(_stats: Dict, hist_label: str, show=True, save=False):

    if hist_label == 'brightness':
        _hist_label = 'brt_hist'
    elif hist_label == 'pixel_value':
        _hist_label = 'pxl_hist'

    _hists = {_l: go.Scatter(
        x=list(range(256)),
        y=(_stats[_l][_hist_label] / _stats[_l].cnt),
        name=_l) for _l in LABELS}

    _img = make_subplots(rows=1, cols=3)

    _img.append_trace(_hists['test'], 1, 1)
    _img.append_trace(_hists['training'], 1, 1)

    _img.append_trace(_hists['training'], 1, 2)
    _img.append_trace(_hists['healthy'], 1, 2)
    _img.append_trace(_hists['any'], 1, 2)

    _img.append_trace(_hists['any'], 1, 3)
    _img.append_trace(_hists['epidural'], 1, 3)
    _img.append_trace(_hists['intraparenchymal'], 1, 3)
    _img.append_trace(_hists['intraventricular'], 1, 3)
    _img.append_trace(_hists['subarachnoid'], 1, 3)
    _img.append_trace(_hists['subdural'], 1, 3)

    _img.update_layout(height=2160, width=3840,
                       title_text=f'Histogram on {hist_label}')

    if show:
        _img.show()

    if save:
        _img.write_image(os.path.join(
            ICH_DIR, f'{hist_label}_histogram.png'))


@click.command()
@click.option('--stats_path', '-i', type=str, default=DEFAULT_STATS_PATH)
@click.option('--debug', '-d', is_flag=True)
def visualize_stats(stats_path: str, debug: bool):
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    trn_stats = Dict({f'{_c}': stats.trn[_c] for _c in CATEGORIES})
    trn_stats.training = deepcopy(trn_stats.all)
    del trn_stats.all
    tst_stats = Dict({'test': stats.tst.all})
    all_stats = Dict({**trn_stats, **tst_stats})

    # Sunburst plot for overall dataset composition
    sunburst_plot(all_stats, save=True)

    # Averaged images
    avg_img(all_stats, save=True)

    histogram(all_stats, hist_label='brightness', save=True)
    histogram(all_stats, hist_label='pixel_value', save=True)


if __name__ == '__main__':
    visualize_stats()

