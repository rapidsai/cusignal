# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 1
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import pkg_resources

from . import _filters
from ..utils.debugtools import print_atts


class KalmanFilter(object):

    """
    This is a multi-point Kalman Filter implementation of
    https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py,
    with a subset of functionality.

    All Kalman Filter matrices are stack on the X axis. This is to allow
    for optimal global accesses on the GPU.

    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u
    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
    dim_u : int (optional)
        Size of the control input, if it is being used.
        Default value of 0 indicates it is not used.
    points : int (optional)
        Number of Kalman Filter points to track.
    dtype : dtype (optional)
        Data type of compute.

    Attributes
    ----------
    x : numpy.array(points, dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.

    P : numpy.array(points, dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.

    z : numpy.array(points, dim_z, 1)
        Last measurement used in update(). Read only.

    R : numpy.array(points, dim_z, dim_z)
        Measurement noise matrix

    Q : numpy.array(points, dim_x, dim_x)
        Process noise matrix

    F : numpy.array(points, dim_x, dim_x)
        State Transition matrix

    H : numpy.array(points, dim_z, dim_x)
        Measurement function

    _alpha_sq : float (points, 1, 1)
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.

    Examples
    --------
    Here is a filter that tracks position and velocity using a sensor that only
    reads position.

    First construct the object with the required dimensionality,
    number of points, and data type.

    .. code::

        import cupy as cp
        import numpy as np

        from cusignal import KalmanFilter

        points = 1024
        kf = KalmanFilter(dim_x=4, dim_z=2, points=points, dtype=cp.float64)

    Assign the initial value for the state (position and velocity)
    for all Kalman Filter points.

    .. code::

        initial_location = np.array(
            [[10.0, 10.0, 0.0, 0.0]], dtype=dt
        ).T  # x, y, v_x, v_y
        kf.x = cp.repeat(
            cp.asarray(initial_location[cp.newaxis, :, :]), points, axis=0
        )

    Define the state transition matrix for all Kalman Filter points:

        .. code::

            F = np.array(
                [
                    [1.0, 0.0, 1.0, 0.0],  # x = x0 + v_x*dt
                    [0.0, 1.0, 0.0, 1.0],  # y = y0 + v_y*dt
                    [0.0, 0.0, 1.0, 0.0],  # dx = v_x
                    [1.0, 0.0, 0.0, 1.0],
                ],  # dy = v_y
                dtype=dt,
            )
            kf.F = cp.repeat(cp.asarray(F[cp.newaxis, :, :]), points, axis=0)

    Define the measurement function for all Kalman Filter points:

        .. code::

            H = np.array(
                [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
                dtype=dt,  # x_0  # y_0
            )
            kf.H = cp.repeat(cp.asarray(H[cp.newaxis, :, :]), points, axis=0)

    Define the covariance matrix for all Kalman Filter points:

        .. code::

            initial_estimate_error = np.eye(dim_x, dtype=dt) * np.array(
                [1.0, 1.0, 2.0, 2.0], dtype=dt
            )
            kf.P = cp.repeat(
                cp.asarray(initial_estimate_error[cp.newaxis, :, :]),
                points,
                axis=0,
            )

    Define the measurement noise  for all Kalman Filter points:

        .. code::

            measurement_noise = np.eye(dim_z, dtype=dt) * 0.01
            kf.R = cp.repeat(
                cp.asarray(measurement_noise[cp.newaxis, :, :]), points, axis=0
            )

    Define the process noise  for all Kalman Filter points:

        .. code::
            motion_noise = np.eye(dim_x, dtype=dt) * np.array(
                [10.0, 10.0, 10.0, 10.0], dtype=dt
            )
            kf.Q = cp.repeat(
                cp.asarray(motion_noise[cp.newaxis, :, :]), points, axis=0
            )

    Now just perform the standard predict/update loop:
    Note: This example just uses the same sensor reading for all points

        .. code::

            kf.predict()
            z = get_sensor_reading()
            kf.z = cp.repeat(z[cp.newaxis, :, :], points, axis=0)
            kf.update()

    Results are in:

        .. code::
            kf.x[:, :, :]

    References
    ----------

    .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
       p. 208-212. (2006)

    .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

    """

    def __init__(
        self, dim_x, dim_z, dim_u=0, points=1, dtype=cp.float32,
    ):

        # Check CuPy version
        ver = pkg_resources.get_distribution("cupy").version
        if ver != "8.0.0b4" or ver != "8.0.0rc1" or ver != "8.0.0":
            raise NotImplementedError(
                "Kalman Filter only compatible with CuPy v.8.0.0b4+"
            )

        self.points = points

        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # Create data arrays
        self.x = cp.zeros((self.points, dim_x, 1,), dtype=dtype)  # state

        self.P = cp.repeat(
            cp.identity(dim_x, dtype=dtype)[cp.newaxis, :, :],
            self.points,
            axis=0,
        )  # uncertainty covariance

        self.Q = cp.repeat(
            cp.identity(dim_x, dtype=dtype)[cp.newaxis, :, :],
            self.points,
            axis=0,
        )  # process uncertainty

        # self.B = None  # control transition matrix

        self.F = cp.repeat(
            cp.identity(dim_x, dtype=dtype)[cp.newaxis, :, :],
            self.points,
            axis=0,
        )  # state transition matrix

        self.H = cp.zeros(
            (self.points, dim_z, dim_z,), dtype=dtype
        )  # Measurement function

        self.R = cp.repeat(
            cp.identity(dim_z, dtype=dtype)[cp.newaxis, :, :],
            self.points,
            axis=0,
        )  # process uncertainty

        self._alpha_sq = cp.ones(
            (self.points, 1, 1,), dtype=dtype
        )  # fading memory control

        self.z = cp.empty((self.points, dim_z, 1,), dtype=dtype)

        # Allocate GPU resources
        threads_z_axis = 16
        d = cp.cuda.device.Device()
        numSM = d.attributes["MultiProcessorCount"]
        threadsperblock = (self.dim_x, self.dim_x, threads_z_axis)
        blockspergrid = (1, 1, numSM * 20)

        max_threads_per_block = self.dim_x * self.dim_x * threads_z_axis

        # Only need to populate cache once
        # At class initialization
        _filters._populate_kernel_cache(
            self.x.dtype,
            threads_z_axis,
            self.dim_x,
            self.dim_z,
            max_threads_per_block,
        )

        # Retrieve kernel from cache
        self.predict_kernel = _filters._get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            _filters.GPUKernel.PREDICT,
        )

        self.update_kernel = _filters._get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            _filters.GPUKernel.UPDATE,
        )

        print_atts(self.predict_kernel)
        print_atts(self.predict_kernel)

    def predict(self):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        """

        self.predict_kernel(
            self._alpha_sq, self.x, self.F, self.P, self.Q,
        )

    def update(self):
        """
        Update Kalman Filter.

        Parameters
        ----------
        """

        self.update_kernel(
            self.x, self.z, self.H, self.P, self.R,
        )
