#!/usr/bin/env python2

from utils import plotting

if __name__ == "__main__":
    plotting.plot_seeg("data/seeg_rec_ext.npz",     'r', "img/seeg_rec_ext.pdf")
    plotting.plot_seeg("results/seeg_sim_Real.npz", 's', "img/seeg_sim_real.pdf")
    plotting.plot_spectral_signatures("data/seeg_rec.npz",         'r', "img/SpectralSignatures_rec.pdf")
    plotting.plot_spectral_signatures("results/seeg_sim_Real.npz", 's', "img/SpectralSignatures_sim_real.pdf")
