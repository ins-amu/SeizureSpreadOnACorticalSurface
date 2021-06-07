#!/usr/bin/env python2

from utils import plotting

if __name__ == "__main__":
    plotting.plot_seizure_evolution("results/Real.npz", "img/SimSeizureEvolution.pdf")
