#!/usr/bin/env python2

"""
Seizure spread simulation on folded cortical surface.

Run as:
python simulate.py <CONFIG_JSON_FILE>
"""


import sys
import os.path
import json

import numpy as np
import scipy.signal as sig

from utils.runsim import runsim

here = os.path.dirname(os.path.abspath(__file__))


def sim(config_file):
    sim_name = os.path.basename(config_file).strip('.json')

    with open(config_file, 'r') as fl:
        config = json.load(fl)
    meshdir = config['meshdir']
    g11 = config['g11']
    g22 = config['g22']
    u0def = config['u0def']
    tave_mon_period = config['tave_mon_period']

    # Stable for x0 = -2.2
    ic = [-1.46242601e+00,  -9.69344913e+00,   2.95029597e+00,  -1.11181819e+00,  -9.56105974e-20,  -4.38727802e-01]
    k = 0.318

    config = {
        'conn_file': os.path.join(meshdir, "connectivity.zip"),
        'surf_file': os.path.join(meshdir, "surface.zip"),
        'regmap_file': os.path.join(meshdir, "region_mapping.txt"),
        'results_dir': os.path.join(here, "results"),
        'gamma11': k*g11, 'gamma12': k*0.1, 'gamma22': k*g22,
        'x0': u0def,
        'locconn_b': 1.0, 'locconn_amp': 1.0,
        'tau0': 20000,
        'tt': 0.17, 'dt': 0.2, 'max_t': 40000,
        'tave_mon': {'period': tave_mon_period},
        'tss_mon': None,
        'seeg_mons': [
            {'period': 2.0,
             'point_file': os.path.join(here, meshdir, 'seeg.txt'),
             'proj_file': os.path.join(here, meshdir, 'gain_mtx_dipole.txt')},
            {'period': 2.0,
             'point_file': os.path.join(here, meshdir, 'seeg.txt'),
             'proj_file': os.path.join(here, meshdir, 'gain_mtx_nearest.txt')}
        ],
        'ic': ic
    }

    if all([os.path.exists(os.path.join(here, meshdir, f)) for f in ["points_pos.txt", "proj_matrix_path.txt"]]):
        config['seeg_mons'].append(
            {'period': 2.0,
             'point_file': os.path.join(here, meshdir, 'points_pos.txt'),
             'proj_file': os.path.join(here, meshdir, 'proj_matrix_path.txt')},
        )

    runsim(sim_name, config)


def get_seeg(sim_name):
    sim_name = os.path.basename(config_file).strip('.json')
    filename = os.path.join("results", sim_name + ".npz")
    results = np.load(filename)
    t = 0.001 * results['t1']     # ms to s
    seeg = results['data1'][:, 0, :, 0].T
    sampling_rate = 1./(t[1] - t[0])
    filtb, filta = sig.butter(3, 0.5 / (0.5 * sampling_rate), 'highpass')
    seeg = sig.filtfilt(filtb, filta, seeg, axis=1)
    names = ["TB%d" % (i+1) for i in range(seeg.shape[0])]

    seeg_filename = os.path.join("results", "seeg_sim_%s.npz" % sim_name)
    np.savez(seeg_filename, t=t, seeg=seeg, names=names)



if __name__ == "__main__":
    config_file = sys.argv[1]

    sim(config_file)
    get_seeg(config_file)
