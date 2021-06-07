import os.path
import shutil
import tempfile
from zipfile import ZipFile

import numpy as np


class StructuralDataset:
    def __init__(self, orientations, areas, centers, cortical, weights_ut, tract_lengths_ut, names):

        nregions = len(names)

        assert orientations.shape == (nregions, 3)
        assert areas.shape == (nregions,)
        assert centers.shape == (nregions, 3)
        assert cortical.shape == (nregions,)
        assert weights_ut.shape == (nregions, nregions)
        assert tract_lengths_ut.shape == (nregions, nregions)

        # Upper triangular -> symmetric matrices
        assert np.sum(np.tril(weights_ut, -1)) == 0
        assert np.sum(np.tril(tract_lengths_ut, -1)) == 0
        self.weights = weights_ut + weights_ut.transpose() - np.diag(np.diag(weights_ut))
        self.tract_lengths = tract_lengths_ut + tract_lengths_ut.transpose() - np.diag(np.diag(tract_lengths_ut))

        self.orientations = orientations
        self.areas = areas
        self.centers = centers
        self.cortical = cortical
        self.names = names


    def save_to_txt_zip(self, filename):

        tmpdir = tempfile.mkdtemp()

        file_areas = os.path.join(tmpdir, 'areas.txt')
        file_orientations = os.path.join(tmpdir, 'average_orientations.txt')
        file_centres = os.path.join(tmpdir, 'centres.txt')
        file_cortical = os.path.join(tmpdir, 'cortical.txt')
        file_weights = os.path.join(tmpdir, 'weights.txt')
        file_tract_lengths = os.path.join(tmpdir, 'tract_lengths.txt')

        np.savetxt(file_areas, self.areas, fmt='%.2f')
        np.savetxt(file_orientations, self.orientations, fmt='%.2f %.2f %.2f')
        np.savetxt(file_cortical, self.cortical, fmt='%d')
        np.savetxt(file_weights, self.weights, fmt='%d')
        np.savetxt(file_tract_lengths, self.tract_lengths, fmt='%.3f')

        with open(file_centres, 'w') as f:
            for i, name in enumerate(self.names):
                f.write('%s %.4f %.4f %.4f\n' % (name, self.centers[i, 0], self.centers[i, 1], self.centers[i, 2]))

        with ZipFile(filename, 'w') as zip_file:
            zip_file.write(file_areas, os.path.basename(file_areas))
            zip_file.write(file_orientations, os.path.basename(file_orientations))
            zip_file.write(file_centres, os.path.basename(file_centres))
            zip_file.write(file_cortical, os.path.basename(file_cortical))
            zip_file.write(file_weights, os.path.basename(file_weights))
            zip_file.write(file_tract_lengths, os.path.basename(file_tract_lengths))

        shutil.rmtree(tmpdir)
