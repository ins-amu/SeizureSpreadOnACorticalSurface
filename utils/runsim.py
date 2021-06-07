#!/usr/bin/env python2

"""
Wrapper around the TVB simulations
"""


import json
import os.path
import sys

import numpy as np
from scipy.optimize import fsolve

from tvb.simulator.lab import *
from tvb.datatypes.equations import SpatialApplicableEquation, FiniteSupportEquation, Gaussian
from tvb.basic.traits import types_basic as basic

from spatepi import SpatEpi


class LaplaceKernel(SpatialApplicableEquation, FiniteSupportEquation):
    """
    A Laplace kernel equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    equation = basic.String(
        label="Laplace kernel",
        default="amp * (1./(2.*b)) * (exp(-abs(var)/b)) + offset",
        locked=True)

    parameters = basic.Dict(
        label="Laplace parameters",
        default={"amp": 1.0, "b": 1.0, "offset": 0.0})


class StepFunction(SpatialApplicableEquation, FiniteSupportEquation):
    """
    A step function.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    equation = basic.String(
        label="Step function",
        default="amp * (where(abs(var) - a <= 0, 1.0, 0.0))",
        locked=True)

    parameters = basic.Dict(
        label="Step function parameters",
        default={"a": 1.0, "amp": 1.0})



def get_equilibrium(model, init):
    nvars = len(model.state_variables)
    cvars = len(model.cvar)

    def func(x):
        fx = model.dfun(x.reshape((nvars, 1, 1)),
                        np.zeros((cvars, 1, 1)))
        return fx.flatten()

    x = fsolve(func, init)
    return x


def run_with_progress(simulator, sim_length):
    PERC_STEP = 1

    output = [([], []) for _ in range(len(simulator.monitors))]

    nextperc = PERC_STEP
    for mon_outputs in simulator(simulation_length=sim_length):
        if mon_outputs[0] is not None:
            time, _ = mon_outputs[0]
            if 100 * time/sim_length >= nextperc:
                print("Simulation progress: %2.0f%%" % nextperc)
                sys.stdout.flush()
                nextperc += PERC_STEP

        for i, mon_output in enumerate(mon_outputs):
            if mon_output is not None:
                output[i][0].append(mon_output[0])
                output[i][1].append(mon_output[1])

    return output



def runsim(name, config):

    default_config = {
        'conn_file': None,
        'surf_file': None,
        'regmap_file': None,
        'results_dir': None,
        'theta11': -1.0, 'theta12': -1.0, 'theta22': -0.5,
        'gamma11': 1.0, 'gamma12': 10.0, 'gamma22': 1.0,
        'stim': None,
        'x0': -2.3,
        'max_t': 3000,
        'dt': 0.02,
        'locconn_b': 1.0, 'locconn_amp': 1.0,
        'Iext1': 3.1, 'Iext2': 0.45, 'tau0': 2857.0, 'tau2': 10.0, 'tau12': 100.0,
        'tt': 1.0,
        'tave_mon': {'period': 1.0},
        'tss_mon': None,
        'seeg_mons': None,
        'ic': None,
        'model': None,
        'noise_vector': None,
        'seed': 0
    }

    for key, val in default_config.items():
        if key not in config:
            config[key] = val

    print('run_sim: %s' % name)

    # Connectome
    con = connectivity.Connectivity.from_file(os.path.abspath(config['conn_file']))
    nregions = len(con.areas)
    con.speed = np.inf

    # Surface and local connectivity kernel
    surf = cortex.Cortex.from_file(source_file=os.path.abspath(config['surf_file']),
                                   region_mapping_file=os.path.abspath(config['regmap_file']))
    loc_conn = local_connectivity.LocalConnectivity(equation=LaplaceKernel(), cutoff=10.0)
    loc_conn.scale_by_area = True
    loc_conn.homogenize = False
    loc_conn.equation.parameters['b'] = config['locconn_b']
    loc_conn.equation.parameters['amp'] = config['locconn_amp']
    loc_conn.surface = surf
    surf.local_connectivity = loc_conn
    surf.configure()
    nverts = surf.number_of_vertices

    # Neural mass model
    epileptors = SpatEpi()
    epileptors.variables_of_interest = ['q1 - u1', 's', 'u1', 'q1', 'u2']

    for var in ['theta11', 'theta22', 'theta12', 'gamma11', 'gamma22', 'gamma12', 'x0',
                'Iext1', 'Iext2', 'tau0', 'tau2', 'tau12', 'tt']:
        value = config[var]
        if type(value) in [int, float, np.int, np.float64]:
            setattr(epileptors, var, value)
        else:
            setattr(epileptors, var, eval(value, {'x': surf.vertices[:, 0],
                                                  'y': surf.vertices[:, 1],
                                                  'z': surf.vertices[:, 2],
                                                  'np': np}))

    # Stimulation
    if config['stim'] is not None:
        stim_x, stim_y, stim_onset, stim_amplitude, stim_duration, stim_radius = config['stim']

        stim_t = equations.PulseTrain()
        stim_t.parameters["onset"] = stim_onset
        stim_t.parameters["tau"] = stim_duration # 10
        stim_t.parameters["T"] = 100000.0

        #stim_s = equations.Gaussian()
        stim_s = StepFunction()
        stim_s.parameters["amp"] = stim_amplitude
        stim_s.parameters["a"] = stim_radius # 0.785

        stim_index = np.argmin(np.sqrt((surf.vertices[:, 0] - stim_x)**2 + (surf.vertices[:, 1] - stim_y)**2))
        stimulus = patterns.StimuliSurface(temporal=stim_t,
                                           spatial=stim_s,
                                           surface=surf,
                                           focal_points_surface=[stim_index])
        stimulus.configure_space()
        stimulus.configure_time(np.arange(0., config['max_t'], config['dt']))
    else:
        stimulus = None

    # Global connectivity
    coupl = coupling.Difference(a=0.)

    # Integration
    if config['noise_vector'] is None:
        heunint = integrators.HeunDeterministic(dt=config['dt'])
    else:
        rand_stream = noise.RandomStream(init_seed=config['seed'])
        add_noise = noise.Additive(nsig=np.array(config['noise_vector']), random_stream=rand_stream)
        heunint = integrators.HeunStochastic(dt=config['dt'], noise=add_noise)

    # Monitors
    mons = []
    if config['tave_mon'] is not None:
        mons.append(monitors.TemporalAverage(period=config['tave_mon']['period']))
    if config['tss_mon'] is not None:
        mons.append(monitors.SubSample(period=config['tss_mon']['period']))
    if config['seeg_mons'] is not None:
        for monconf in config['seeg_mons']:
            mons.append(monitors.iEEG.from_file(sensors_fname=monconf['point_file'],
                                                projection_fname=monconf['proj_file'],
                                                period=monconf['period']))

    # Initial conditions
    if config['ic'] is None:
        epileptor = SpatEpi()
        epileptor.x0 = -2.2
        ic = get_equilibrium(epileptor, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
    else:
        ic = config['ic']

    print("IC: %s" % ic)
    ic_full = np.repeat(ic, nverts).reshape((1, len(ic), nverts, 1))

    # Simulator
    sim = simulator.Simulator(model=epileptors,
                              connectivity=con,
                              coupling=coupl,
                              integrator=heunint,
                              monitors=mons,
                              initial_conditions=ic_full,
                              stimulus=stimulus,
                              surface=surf)
    sim.configure()

    # Run
    results = run_with_progress(sim, config['max_t'])

    resultsdir = config['results_dir']
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)

    if len(results) == 1:
        (t, data), = results
        np.savez(os.path.join(resultsdir, "%s.npz" % name), t=t, data=data, x0=epileptors.x0,
                 coordx=surf.vertices[:, 0], coordy=surf.vertices[:, 1], coordz=surf.vertices[:, 2])
    else:
        results_dict = {
            'coordx': surf.vertices[:, 0],
            'coordy': surf.vertices[:, 1],
            'coordz': surf.vertices[:, 2],
            'x0': epileptors.x0,
        }
        for i, (t, data) in enumerate(results):
            results_dict['t%d' % i] = t
            results_dict['data%d' % i] = data
        np.savez(os.path.join(resultsdir, "%s.npz" % name), **results_dict)

    with open(os.path.join(resultsdir, "%s.json" % name), 'w') as fl:
        json.dump(config, fl, indent=4, sort_keys=True)
