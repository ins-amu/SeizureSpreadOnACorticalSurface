
import json
import os
import re
from zipfile import ZipFile

from mayavi import mlab
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec

import numpy as np
import mne

import scalebars
from plotutils import add_panel_letters, add_scalebar

from contacts import Contacts
from seeg import make_bipolar


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


def plot_seeg_fig(ax1, ax2, data, detail, scaling, snapshots=None, bipolar=True, selected_contacts=None,
                  scalebar_size=None, scalebar_label=None):
    t, signals, names_mon = data

    if bipolar:
        seeg, names = make_bipolar(signals, names_mon)
    else:
        seeg, names = signals, names_mon

    names = list(names)
    if selected_contacts:
        inds = [names.index(name) for name in selected_contacts]
        seeg = seeg[inds, :]
        names = selected_contacts

    ns = len(names)

    if snapshots is None:
        snapshots = []

    # Whole seizure
    plt.sca(ax1)
    for i in range(ns):
        color = 'g' if names[i][:1] == "B" else 'b'
        plt.plot(t, scaling*seeg[i, :] + i, '-', color=color, lw=0.25)
    plt.yticks(np.r_[:ns], names)
    plt.ylim([-1.0, ns + 0.0])

    for label, tt in snapshots:
        plt.axvline(tt, color='g', lw=1)
        plt.text(tt + 0.2, ns - 0.4, label, fontsize=12, color='g')

    plt.xlabel('t [s]')
    scalebar_label = scalebar_label if scalebar_label is not None else str(scalebar_size)
    if scalebar_size is not None:
        sb1 = scalebars.AnchoredScaleBar(ax1.transData, sizey=scalebar_size*scaling, labely=scalebar_label,
                                         pad=0.5, barwidth=2, sep=10, textprops=dict(size=10), loc=2)
        ax1.add_artist(sb1)

    if (ax2 is not None) and (detail is not None):
        # Detail
        rect = matplotlib.patches.Rectangle((detail[0], -1+0.01), detail[1] - detail[0], ns + 1 - 0.02,
                                            linewidth=1, edgecolor='0.5',facecolor='none', clip_on=False)
        ax1.add_patch(rect)

        # Onset detail
        plt.sca(ax2)
        plt.xlim(detail)
        for i in range(ns):
            color = 'g' if names[i][:1] == "B" else 'b'
            plt.plot(t, scaling*seeg[i, :] + i, '-', color=color, lw=0.25)
        plt.yticks(np.r_[:ns], names)
        plt.ylim([-1.0, ns + 0.0])
        plt.xlabel('t [s]')
        if scalebar_size is not None:
            sb2 = scalebars.AnchoredScaleBar(ax2.transData, sizey=scalebar_size*scaling, labely=scalebar_label,
                                             pad=0.5, barwidth=2, sep=10, textprops=dict(size=10), loc=2)
            ax2.add_artist(sb2)


def plot_seeg(filename, source, out_file):
    if not os.path.isfile(filename):
        print("File '%s' does not exist, skipping" % filename)
        return

    with np.load(filename) as data:
        t = data['t']
        seeg = data['seeg']
        names = data['names']

    # fig = plt.figure(figsize=(22.5, 7))
    fig = plt.figure(figsize=(7.5, 2.3))

    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)

    if source == 'r':
        plot_seeg_fig(ax1, ax2, (t, seeg, names),
                      detail=(20., 40.), scaling=200., scalebar_size=0.005, scalebar_label="5 mV")
    elif source == 's':
        plot_seeg_fig(ax1, ax2, (t, seeg, names),
                      detail=(4., 15.), scaling=0.2, scalebar_size=3, scalebar_label="3 a.u.")

    add_panel_letters(fig, fontsize=16)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.16, wspace=0.4, hspace=0.0)
    plt.savefig(out_file)
    plt.close()



def plot_spectral_signatures(filename, source, out_file):
    MINDS = [4, 5, 6]
    BINDS = [4, 5]

    if not os.path.isfile(filename):
        print("File '%s' does not exist, skipping" % filename)
        return

    if source == 's':
        scaling = 0.2
        tlim = (6, 18)
        scalebar_label = "2 a.u."
        scalebar_size = 2.0
        vmin, vmax = -5, 35
    elif source == 'r':
        scaling = 300.0
        tlim = (24, 36)
        scalebar_label = "2 mV"
        scalebar_size = 0.002
        vmin, vmax = -60, -20

    with np.load(filename) as data:
        t = data['t']
        seeg_mop = data['seeg']
        names_mop = data['names']

    seeg_bip, names_bip = make_bipolar(seeg_mop, names_mop)
    sfreq = 1./(t[1] - t[0])
    freqs = np.logspace(np.log10(2), np.log10(100), 50)
    sxx_ = mne.time_frequency.tfr_array_morlet(np.expand_dims(seeg_bip, 0),
                                               sfreq=sfreq,
                                               freqs=freqs,
                                               n_cycles=9,
                                               zero_mean=False, output='avg_power')

    sxx = np.zeros_like(sxx_)
    for i in range(sxx_.shape[0]):
        sxx[i] = (sxx_[i].T * freqs).T

    fig = plt.figure(figsize=(5, 2.4))

    # Traces
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    for i, ind in enumerate(MINDS):
        plt.plot(t, scaling * seeg_mop[ind] + i, 'b-', lw=0.3)
    plt.axhline(-0.5, ls='--', color='0.5')

    for i, ind in enumerate(BINDS):
        plt.plot(t, scaling * seeg_bip[ind] - 2 + i, 'b-', lw=0.3)

    plt.yticks(np.r_[-len(BINDS):len(MINDS)],
               [names_bip[ind] for ind in BINDS] + [names_mop[ind] for ind in MINDS])
    plt.xlim(tlim)
    plt.ylim([-2.5, 2.5])
    plt.xlabel("t [s]")
    add_scalebar(ax1, label=scalebar_label, size=scalebar_size, scaling=scaling, loc=(0.0, 0.7), fontsize=10)


    # Bipolar TFA
    for i, bind in enumerate(BINDS):
        ax2 = plt.subplot2grid((2, 2), (1-i, 1))
        im = plt.imshow(10 * np.log10(sxx[bind]),
                        aspect='auto', origin='lower',
                        extent=[t[0], t[-1], 0, len(freqs)],
                        vmin=vmin, vmax=vmax,
                        cmap='jet')

        plt.xlim(tlim)
        plt.yticks(np.r_[:len(freqs):11], ["%4.1f" % f for f in freqs[::11]])
        plt.ylabel("f [Hz]")
        plt.text(0.07, 0.85, names_bip[bind], transform=ax2.transAxes,
                 bbox={'facecolor': 'white'})

        if i == 1:
            plt.xticks([])
        else:
            plt.xlabel("t [s]")

    plt.subplots_adjust(bottom=0.17, top=0.87, left=0.08, right=0.88, wspace=0.4, hspace=0.15)
    cbar_ax = fig.add_axes([0.89, 0.18, 0.015, 0.67])
    plt.colorbar(im, cax=cbar_ax, label='dB')
    add_panel_letters(fig, axes=[ax1, ax2], ypos=[1.05, 1.1], fontsize=16)
    plt.savefig(out_file)



def get_3d_fig(verts, triangs, scalar, figsize, colormap, vmin, vmax, colorbar=False,
               contact_pos=None, contact_labels=None, orientation_axes_pos=None, view=None):

    fig = mlab.figure(size=figsize, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    wire = mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2],
                                triangs, representation='wireframe',
                                color=(0, 0, 0), line_width=1.0, figure=fig)

    colormap_is_table = type(colormap) == np.ndarray

    surf = mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], triangs,
                                scalars=scalar, representation='surface',
                                colormap=(colormap if not colormap_is_table else 'coolwarm'),
                                figure=fig, vmin=vmin, vmax=vmax, opacity=1.0)
    if colormap_is_table:
        surf.module_manager.scalar_lut_manager.lut.table = colormap

    if colorbar:
        mlab.colorbar(surf, orientation='vertical', nb_labels=0)

    if contact_pos is not None:
        mlab.points3d(contact_pos[:, 0], contact_pos[:, 1], contact_pos[:, 2],
                      name="contacts", scale_factor=1, figure=fig, color=(1, 1, 0))
        if contact_labels is not None:
            for i in range(contact_pos.shape[0]):
                mlab.text3d(contact_pos[i, 0], contact_pos[i, 1], contact_pos[i, 2], contact_labels[i],
                            color=(1, 1, 0), scale=1.6, figure=fig)

    if view is None:
        view = dict(azimuth=240, elevation=10, distance=90.0)
    mlab.view(**view)

    if orientation_axes_pos is not None:
        oa = mlab.orientation_axes(xlabel="", ylabel="", zlabel="")
        oa.marker.set_viewport(*orientation_axes_pos)


    return fig


def scalar_img(verts, triangs, scalar, figsize, colormap, vmin, vmax, colorbar=False,
               contact_pos=None, contact_labels=None, orientation_axes_pos=None, view=None):
    fig = get_3d_fig(verts, triangs, scalar, figsize, colormap, vmin, vmax, colorbar, contact_pos,
                     contact_labels, orientation_axes_pos, view)
    img = mlab.screenshot(fig)
    mlab.close()
    return img


def plot_seizure_evolution(filename, out_file):

    results = np.load(filename)
    with open(os.path.splitext(filename)[0] + ".json", 'r') as fl:
        params = json.load(fl)

    if 'data' in results.keys():
        t = 0.001 * results['t']
        data = results['data']
    else:
        t = 0.001 * results['t0']
        data = results['data0']

    nverts = data.shape[2]
    q1mu1 = data[:, 0, :, 0]
    s     = data[:, 1, :, 0]
    u1    = data[:, 2, :, 0]
    q1    = data[:, 3, :, 0]

    contacts = Contacts(os.path.join(os.path.dirname(params["conn_file"]), "seeg.txt"))
    with ZipFile(params['surf_file']) as zf:
        with zf.open("vertices.txt", "r") as fl:
            verts = np.genfromtxt(fl, dtype=float)
        with zf.open("triangles.txt", "r") as fl:
            triangs = np.genfromtxt(fl, dtype=int)


    fig = plt.figure(figsize=(7, 3.33))

    gs1 = gridspec.GridSpec(1, 6, left=0.01, right=0.97, top=0.99, bottom=0.66, wspace=0)
    gs2 = gridspec.GridSpec(1, 3, left=0.16, right=0.97, top=0.59, bottom=0.43)
    gs3 = gridspec.GridSpec(1, 6, left=0.01, right=0.97, top=0.34, bottom=0.00, wspace=0)

    axes = []

    # First row: seizure extent
    colormap = (255*matplotlib.cm.get_cmap('jet')(np.linspace(0., 1., 256))).astype(int)
    for i, tt in enumerate([5, 10, 15, 20, 30, 39.99]):
        ax = plt.subplot(gs1[0, i])
        tind = np.where(t >= tt)[0][0]
        ax.imshow(scalar_img(verts, triangs,
                             np.where(u1[tind, :] > -0.8, 1.0, 0.0), vmin=-0.45, vmax=1.25,
                             figsize=(1500, 1500),
                             colormap=colormap,
                             contact_pos=contacts.xyz),
                  interpolation='none')
        ax.axis('off')
        ax.text(0.6, 0.1, '%2.0f s' % tt, fontsize=8, transform=ax.transAxes)
        if i == 0:
            axes.append(ax)
        if i == 4:
            plt.axvline(0, ymin=0.05, ymax=0.95, color='k', lw=2)

    # Last row: fast waves
    colormap = (255*matplotlib.cm.get_cmap('coolwarm')(np.linspace(0., 1., 256))).astype(int)
    for i, tt in enumerate([19.96, 19.98, 20.00, 20.02, 20.04, 20.06]):
        ax = plt.subplot(gs3[0, i])
        tind = np.where(t > tt)[0][0]
        ax.imshow(scalar_img(verts, triangs,
                             q1mu1[tind, :], vmin=-2.0, vmax=1.4,
                             figsize=(1500, 1500),
                             colormap=colormap,
                             contact_pos=contacts.xyz,
                             #contact_labels=contacts.names,
                             ),
                  interpolation='none')
        ax.axis('off')
        ax.text(0.6, 0.1, '%5.2f s' % tt, fontsize=8, transform=ax.transAxes)

        if i == 0:
            axes.append(ax)

    # Middle row: source time series
    t = 0.001 * results['t2']
    data = results['data2'][:, 0, :9, 0].T
    names = ["TB%d" % (i+1) for i in range(data.shape[0])]
    scaling = 0.2

    for i, tlim in enumerate([(5, 7), (19, 21), (30, 32)]):
        ax = plt.subplot(gs2[0, i])
        tmask = (t >= tlim[0]) * (t < tlim[1])
        for j, ind in enumerate([0, 2]):
            plt.plot(t[tmask], scaling * data[ind, tmask] + j, color='b', lw=0.5)
        plt.yticks([])
        xtpos = range(int(tlim[0] - 1), int(tlim[1]) + 1)
        plt.xticks(xtpos, ["%2d s" % a for a in xtpos], fontsize=8)
        plt.ylim([-1, 1.5])
        plt.xlim(tlim)
        for pos in ['left', 'right', 'top']:
            ax.spines[pos].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        if i == 1:
            rect = matplotlib.patches.Rectangle((19.94, -0.6), 0.16, 2.0, ec='g', fc='none', lw=1)
            ax.add_patch(rect)
            plt.plot([20.1, 20.15, 20.15], [-0.5, -0.5, -2.5], 'g-', lw=1, clip_on=False)
            rect2 = matplotlib.patches.Rectangle((15.34, -7.6), 8.18, 5.1, ec='g', fc='none', lw=2, clip_on=False)
            ax.add_patch(rect2)
            rect3 = matplotlib.patches.Rectangle((15.34,  2.6), 8.18, 5.1, ec='b', fc='none', lw=2, clip_on=False)
            ax.add_patch(rect3)

        if i == 0:
            axes.insert(1, ax)

    # Lines from A1
    l1 = matplotlib.lines.Line2D((0.060, 0.060, 0.150), (0.768, 0.50, 0.50), transform=fig.transFigure, color='k', lw=1)
    l2 = matplotlib.lines.Line2D((0.075, 0.075, 0.150), (0.790, 0.56, 0.56), transform=fig.transFigure, color='k', lw=1)
    fig.lines = l1, l2


    add_panel_letters(fig, axes, fontsize=16, xpos=[0.1, 0.0, 0.1], ypos=[0.8, 0.95, 0.8])
    # plt.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.04, hspace=0.2)
    plt.savefig(out_file)
    plt.close()
