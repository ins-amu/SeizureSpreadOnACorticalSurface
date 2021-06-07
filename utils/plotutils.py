# coding=utf-8

from __future__ import unicode_literals
import itertools
import string

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np

import scalebars


def add_panel_letters(fig, axes=None, fontsize=30, xpos=-0.04, ypos=1.05):
    if axes is None:
        axes = fig.get_axes()

    if type(xpos) == float:
        xpos = itertools.repeat(xpos)
    if type(ypos) == float:
        ypos = itertools.repeat(ypos)

    for i, (ax, x, y) in enumerate(zip(axes, xpos, ypos)):
        ax.text(x, y, string.ascii_uppercase[i],
                transform=ax.transAxes, size=fontsize, weight='bold')


"""
765
804
123
"""
annot_args = {
    '0': {'ha': 'center', 'va': 'center'},
    '1': {'ha': 'right',  'va': 'top'},
    '2': {'ha': 'center', 'va': 'top'},
    '3': {'ha': 'left',   'va': 'top'},
    '4': {'ha': 'left',   'va': 'center'},
    '5': {'ha': 'left',   'va': 'bottom'},
    '6': {'ha': 'center', 'va': 'bottom'},
    '7': {'ha': 'right',  'va': 'bottom'},
    '8': {'ha': 'right',  'va': 'center'}
}

def annotate_scatter(scat, labels, poss, fontsize=8):
    for xy, label, pos in zip(scat.get_offsets(), labels, poss):
        plt.annotate(label, xy, xy, fontsize=fontsize, **annot_args[pos])


def corrcoeff_box(ax, x, y, loc, fontsize=12):
    cc = np.corrcoef(x, y)[0, 1]
    plt.text(loc[0], loc[1], "rho = %.2f" % (cc), fontsize=fontsize,
             va='center', ha='center', transform=ax.transAxes,
             bbox={'boxstyle': 'square', 'facecolor': 'white'})


def add_scalebar(ax, label, size, scaling, loc, fontsize=14):
    sb = scalebars.AnchoredScaleBar(ax.transData,
                                    sizey=size*scaling,
                                    labely=label,
                                    pad=1, barwidth=2, sep=10, textprops=dict(size=fontsize),
                                    bbox_to_anchor=loc,
                                    bbox_transform=ax.transAxes,
                                    loc=3)
    ax.add_artist(sb)


def add_inset_colorbar(ax):
    cbaxes = inset_axes(ax, width="25%", height="5%", loc=3, borderpad=2.0)
    cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()
    cbaxes.xaxis.set_ticks_position('top')
    cbaxes.xaxis.set_label_position('top')


def set_corner_pixels(image, x_min, x_max, y_min, y_max):
    x_diff = x_max-x_min
    y_diff = y_max-y_min

    y_size, x_size = image.get_size()

    x_ppx = x_diff/(x_size-1)
    y_ppx = y_diff/(y_size-1)

    extents = [
        x_min - 0.5*x_ppx,
        x_max + 0.5*x_ppx,
        y_min - 0.5*y_ppx,
        y_max + 0.5*y_ppx
    ]

    image.set_extent(extents)
