"""
Generate 2D rectangular surface
"""

import os
import sys
import shutil
import tempfile
import zipfile

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from structural_dataset import StructuralDataset

def generate_vertices(nx, ny, xlim, ylim):
    xvs, yvs, zvs = np.meshgrid(np.linspace(xlim[0], xlim[1], nx), np.linspace(ylim[0], ylim[1], ny), [0.0])
    xs = xvs.flatten()
    ys = yvs.flatten()
    zs = zvs.flatten()
    vertices = np.column_stack([xs, ys, zs])
    return vertices


def generate_triangles(nx, ny):
    triangles = np.empty((2 * (nx - 1) * (ny - 1), 3), dtype=int)
    for j in range(ny - 1):
        for i in range(nx - 1):
            p1 = j * nx + i
            p2 = p1 + 1
            p3 = p1 + nx
            p4 = p3 + 1
            triangles[2 * (j * (nx - 1) + i), :] = [p1, p3, p2]
            triangles[2 * (j * (nx - 1) + i) + 1, :] = [p2, p3, p4]

    return triangles


def generate_region_mapping(vertices):
    regmap = np.zeros(vertices.shape[0], dtype=int)
    regmap[vertices[:, 0] >= np.mean(vertices[:, 0])] = 1

    return regmap


def generate_normals(vertices):
    normals = np.zeros((vertices.shape[0], 3))
    normals[:, 2] = 1.0
    return normals


def save_surf(name, vertices, triangles, normals):
    tmpdir = tempfile.mkdtemp()
    file_vertices = os.path.join(tmpdir, 'vertices.txt')
    file_triangles = os.path.join(tmpdir, 'triangles.txt')
    file_normals = os.path.join(tmpdir, 'normals.txt')

    np.savetxt(file_vertices, vertices, fmt='%.6f %.6f %.6f')
    np.savetxt(file_triangles, triangles, fmt='%d %d %d')
    np.savetxt(file_normals, normals, fmt='%.6f %.6f %.6f')

    with zipfile.ZipFile(name, 'w') as zip_file:
        zip_file.write(file_vertices, os.path.basename(file_vertices))
        zip_file.write(file_triangles, os.path.basename(file_triangles))
        zip_file.write(file_normals, os.path.basename(file_normals))

    shutil.rmtree(tmpdir)


def save_conn(name, vertices, regmap):
    regions = np.unique(regmap)
    nreg = len(regions)
    areas = np.ones(nreg)  # This is false, but it's not used for anything anyway
    centres = np.zeros((nreg, 3))
    for reg in regions:
        centres[reg, :] = np.mean(vertices[regmap == reg, :], axis=0)
    orientations = np.zeros((nreg, 3))
    orientations[:, 2] = 1.0
    cortical = np.ones(nreg, dtype=int)
    lengths = np.zeros((nreg, nreg))
    weights = np.zeros((nreg, nreg))
    names = ["Region-%d" % i for i in range(nreg)]

    dataset = StructuralDataset(orientations, areas, centres, cortical, weights, lengths, names)
    dataset.save_to_txt_zip(name)


def plot_surf(filename, vertices, triangles, regmap):
    plt.figure(figsize=(20, 10))

    segments = []

    for triangle in triangles:
        points = [vertices[triangle[i]] for i in range(3)]

        for p1, p2 in [[0, 1], [1, 2], [0, 2]]:
            segments.append(((points[p1][0], points[p1][1]), (points[p2][0], points[p2][1])))

    linecoll = matplotlib.collections.LineCollection(segments, colors='k', lw=0.3)
    plt.gca().add_collection(linecoll)

    plt.scatter(vertices[:, 0], vertices[:, 1], c=regmap, s=10, zorder=10)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.gca().set_aspect('equal')
    plt.savefig(filename)
    plt.close()


def create_2d_surf(datadir, name, nx, ny, xlim, ylim):

    vertices = generate_vertices(nx, ny, xlim, ylim)
    triangles = generate_triangles(nx, ny)
    regmap = generate_region_mapping(vertices)
    normals = generate_normals(vertices)

    save_conn(os.path.join(datadir, 'conn_%s.zip' % name), vertices, regmap)
    save_surf(os.path.join(datadir, 'surf_%s.zip' % name), vertices, triangles, normals)
    np.savetxt(os.path.join(datadir, 'regmap_%s.txt' % name), regmap, fmt='%d')
    plot_surf(os.path.join(datadir, 'fig_%s.png' % name), vertices, triangles, regmap)
