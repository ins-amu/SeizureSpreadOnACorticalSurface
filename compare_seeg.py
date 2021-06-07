#!/usr/bin/env python2

import os

import numpy as np

from utils.analysis import Results


def compare_seeg(target_filename, filenames):
    print("=================================================================")
    print("Comparison with %s" % os.path.basename(target_filename))
    print("----------------------------------------------------")
    print("                      Name       RMSE    MAE  EM-RHO")
    res_trg = Results(target_filename)
    for filename in filenames:
        res = Results(filename)
        rmse = np.sqrt(np.mean((res.onset_times - res_trg.onset_times)**2))
        mae = np.mean(np.abs(res.onset_times - res_trg.onset_times))
        envmean_corrcoeff = np.corrcoef(res.envelope_mean, res_trg.envelope_mean)[0, 1]

        print("%30s %6.2f %6.2f %7.2f" % (os.path.basename(filename), rmse, mae, envmean_corrcoeff))
    print("=================================================================")


def compare_gain(filename, gain_mtx_file):
    res = Results(filename)
    gain_mtx = np.genfromtxt(gain_mtx_file)
    gdiff_norm = np.abs(np.sum(gain_mtx[1:, :] - gain_mtx[:-1, :], axis=1))

    rho = np.corrcoef(res.envelope_mean, gdiff_norm)[0, 1]
    print("=================================================================")
    print("Corrcoef %s <-> %s: %6.2f" % (os.path.basename(filename), os.path.basename(gain_mtx_file), rho))
    print("=================================================================")


if __name__ == "__main__":
    compare_seeg("results/seeg_sim_Real.npz"
                 ["results/seeg_sim_S-Realistic.npz", "results/seeg_sim_S-Flat.npz", "results/seeg_sim_S-Sine.npz"])
    compare_gain("results/seeg_sim_Real.npz", "data/Real/gain_mtx_dipole.txt")
