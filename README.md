

# Computational modeling of seizure spread on a cortical surface

This repository contains the code to perform the main steps of the simulation and analysis of a seizure spread on a cortical surface as described in [1].



## Environment

The simulation and analysis is implemented in Python 2.7 using wide range of packages. Get it ready using the supplied Anaconda environment file
```
    conda env create -f environment.yml
```
which creates the environment named `taa`. The environment can be activated by command
```
    source activate taa
```

The simulations itself are run using The Virtual Brain in version 1.5.4 with several specific patches. The easiest way to get everything running is the so-called [unaided contributor setup](http://docs.thevirtualbrain.org/manuals/ContributorsManual/ContributorsManual.html#the-unaided-setup)  with the patched version of TVB (https://github.com/sipv/tvb-library/tree/surf-sim). In short, run the following commands:
```
  source activate taa                                 # Activate the project environment
  cd /PATH/TO/TVB/DIRECTORY                           # Use directory where you want to install the TVB
  git clone https://github.com/sipv/tvb-library/      # Clone the repository
  cd tvb-library
  git checkout v1.4.0_surf-sim-1.0                    # Checkout the specific version
  python setup.py develop                             # Install it in the project environment
```

## Simulations of the seizure spread on a folded cortical surface

The directory `FoldedSurfaceSimulation/` contains the data and code necessary to simulate the seizure spread on a cortical surface. In the `data/` directory, four subdirectories contain the four surfaces on which the simulations are performed. In each subdirectory, following files are present: 

  * `connectivity.zip`: connectivity in TVB format
  * `surface.zip`: triangulated surface in TVB format
  * `region_mapping.txt`: mapping of the surface vertices to regions
  * `gain_mtx_dipole.txt`: gain matrix using the dipole model
  * `gain_mtx_nearest.txt`: dummy gain matrix for projection of the activity from the nearest vertex.

  For a detailed description of the TVB data format, see the [TVB documentation](http://docs.thevirtualbrain.org/manuals/UserGuide/Complete_Dataset_Description.html).

The `simConfig/` directory contains the configuration files for the simulations.

To run the simulations and analysis, follow these steps:

1. Activate the environment.
```
  cd <PROJECT_DIR>
  source activate taa
  export PYTHONPATH=$PWD:$PYTHONPATH
```
2. Run the simulations:
```
  ./run_simulations.sh
```
Note that the simulations are computationally expensive both in time and in memory required - make sure you have enough RAM available (> 20 GB) before running the simulations.
The successful run will produce a result file in `results/` directory and a separate file with the simulated SEEG in the same place.

3. Now we can plot and analyze the results. 
First, plot the seizure evolution on a 3D cortex (Fig. 4).
```
  python plot_sim_results.py
```
Next, plot the simulated SEEG signals (Fig. 5 in the paper) and spectrogram (Fig. 8).
```
  python plot_seeg.py
```
Finally, calculate the onset times and envelope means, and evaluate the fit (Tab. 2). The recorded patient SEEG is not available in the dataset, so the script compares two simulated SEEG signals instead.
```
  python compare_seeg.py
```

## License

This work is licensed under MIT license. See LICENSE.txt for the full text.

## Authors

Viktor Sip, Maxime Guye, Fabrice Bartolomei, Viktor Jirsa

Aix Marseille Université (AMU) / Centre National de la Recherche Scientifique (CNRS) / Institut National de la Santé et de la Recherche Médicale (INSERM) / Assistance Publique - Hôpitaux de Marseille (AP-HM)

The authors wish to acknowledge the financial support of the following agencies: the French National Research Agency (ANR) as part of the second "Investissements d'Avenir" program (ANR-17-RHUS-0004, EPINOV), European Union's Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 785907, PHRC-I 2013 EPISODIUM (grant number 2014-27), the Fondation pour la Recherche Médicale (DIC20161236442) and the SATT Sud-Est (827-SA-16-UAM) for providing funding for this research project.

## References

[1] Sip, V., Guye, M., Bartolomei, F., Jirsa, V.: _Computational modeling of seizure spread on a cortical surface_. bioRxiv (2020). https://doi.org/10.1101/2020.12.22.423957
