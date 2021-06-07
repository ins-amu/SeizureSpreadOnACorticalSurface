# -*- coding: utf-8 -*-
#

"""
Spatially extended Epileptor model.

"""


from tvb.simulator.models.base import ModelNumbaDfun, LOG, numpy, basic, arrays
from numba import guvectorize, float64, jit

@guvectorize([(float64[:],) * 14], '(n),(m)' + ',()'*11 + '->(n)', nopython=True, target='cpu')
def _numba_dfun(y, c_pop, x0, Iext, Iext2, loc11, loc22, loc12, tt, y0, tau0, tau2, gamma, ydot):
    "Gufunc for Epileptor model equations."

    # population 1
    if y[0] < 0.0:
        ydot[0] = y[0]**3 - 3 * y[0]**2
    else:
        ydot[0] = (y[3] - 0.6 * (y[2] - 4.0) ** 2) * y[0]

    ydot[0] = tt[0] * (y[1] - ydot[0] - y[2] + Iext[0] + loc11[0] + c_pop[0])
    ydot[1] = tt[0] * (y0[0] - 5*y[0]**2 - y[1])

    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    ydot[2] = tt[0] * (1.0/tau0[0] * (4.0 * (y[0] - x0[0]) - y[2] + ydot[2]))

    # population 2
    ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + 2 * y[5] - 0.3 * (y[2] - 3.5) + loc22[0])
    if y[3] < -0.25:
        ydot[4] = 0.0
    else:
        ydot[4] = 6.0 * (y[3] + 0.25)
    ydot[4] = tt[0] * ((-y[4] + ydot[4]) / tau2[0])

    # filter
    ydot[5] = tt[0] * (-0.01 * y[5] + 0.003 * y[0] + 0.01 * loc12[0])


class SpatEpi(ModelNumbaDfun):
    _ui_name = "SpatEpi"
    ui_configurable_parameters = []

    y0 = arrays.FloatArray(
        label="y0",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable",
        order=-1)

    tau0 = arrays.FloatArray(
        label="tau0",
        default=numpy.array([2857.0]),
        doc="Temporal scaling in the third state variable",
        order=4)

    tau2 = arrays.FloatArray(
        label="tau2",
        default=numpy.array([10.0]),
        doc="Temporal scaling in the fifth state variable",
        order=4)

    x0 = arrays.FloatArray(
        label="x0",
        range=basic.Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter",
        order=3)

    Iext = arrays.FloatArray(
        label="Iext",
        range=basic.Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population",
        order=1)

    Iext2 = arrays.FloatArray(
        label="Iext2",
        range=basic.Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population",
        order=2)

    gamma = arrays.FloatArray(
        label="gamma",
        default=numpy.array([0.01]),
        doc="Temporal integration scaling"
    )

    gamma11 = arrays.FloatArray(
        label="gamma11",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 1-1"
    )

    gamma22 = arrays.FloatArray(
        label="gamma22",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 2-2"
    )

    gamma12 = arrays.FloatArray(
        label="gamma12",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 1-2"
    )

    gamma_glob = arrays.FloatArray(
        label="gamma_glob",
        default=numpy.array([1.0]),
        doc="Scaling of the global connections"
    )

    theta11 = arrays.FloatArray(
        label="theta11",
        default=numpy.array([-1.1]),
        doc="Firing threshold 1-1"
    )

    theta22 = arrays.FloatArray(
        label="theta22",
        default=numpy.array([-0.5]),
        doc="Firing threshold 2-2"
    )

    theta12 = arrays.FloatArray(
        label="theta12",
        default=numpy.array([-1.1]),
        doc="Firing threshold 1-2"
    )

    tt = arrays.FloatArray(
        label="tt",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system",
        order=9)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"u1": numpy.array([-2., 1.]),
                 "u2": numpy.array([-20., 2.]),
                 "s": numpy.array([2.0, 5.0]),
                 "q1": numpy.array([-2., 0.]),
                 "q2": numpy.array([0., 2.]),
                 "g": numpy.array([-1., 1.])},
        doc="Typical bounds on state variables in the Epileptor model.",
        order=16
        )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=['u1', 'u2', 's', 'q1', 'q2', 'g', 'q1 - u1'],
        default=['q1 - u1', 's'],
        select_multiple=True,
        doc="Quantities of the Epileptor available to monitor.",
        order=100
    )

    state_variables = ['u1', 'u2', 's', 'q1', 'q2', 'g']

    _nvar = 6
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, x, c, local_coupling=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T

        if type(local_coupling) == float:
            loc11 = self.gamma11 * local_coupling * (0.5 * (numpy.sign(x[0, :, 0] - self.theta11) + 1.0))
            loc22 = self.gamma22 * local_coupling * (0.5 * (numpy.sign(x[3, :, 0] - self.theta22) + 1.0))
            loc12 = self.gamma12 * local_coupling * (0.5 * (numpy.sign(x[0, :, 0] - self.theta12) + 1.0))
        else:
            loc11 = self.gamma11 * local_coupling.dot(0.5 * (numpy.sign(x[0, :, 0] - self.theta11) + 1.0))
            loc22 = self.gamma22 * local_coupling.dot(0.5 * (numpy.sign(x[3, :, 0] - self.theta22) + 1.0))
            loc12 = self.gamma12 * local_coupling.dot(0.5 * (numpy.sign(x[0, :, 0] - self.theta12) + 1.0))

        deriv = _numba_dfun(x_, self.gamma_glob * c_,
                            self.x0, self.Iext, self.Iext2,
                            loc11, loc22, loc12,
                            self.tt, self.y0,
                            self.tau0, self.tau2, self.gamma)
        return deriv.T[..., numpy.newaxis]
