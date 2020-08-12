# Test a kalman filter class

import cupy as cp
import numpy as np
import filterpy.kalman
import cusignal
import time
import itertools
from cupy import prof

dim_x = 4
dim_z = 4
loops = 10
iterations = 250
minimum = 1024

cpu_baseline32 = 0.0
cpu_baseline64 = 0.0

debug = None
test = None


def main(dt, num_points):
    print("num_points", num_points)
    print("iterations", iterations)
    print("data type", dt)
    print("loops", loops)

    cuS = cusignal.KalmanFilter(dim_x, dim_z, points=num_points, dtype=dt)

    f_fpy = filterpy.kalman.KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # State Space Equations
    F = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],  # x = x0 + v_x*dt
            [0.0, 1.0, 0.0, 1.0],  # y = y0 + v_y*dt
            [0.0, 0.0, 1.0, 0.0],  # dx = v_x
            [1.0, 0.0, 0.0, 1.0],
        ],  # dy = v_y
        dtype=dt,
    )

    # Observability Input
    if dim_z == 2:
        H = np.array(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            dtype=dt,  # x_0  # y_0
        )
    elif dim_z == 3:
        H = np.array(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=dt,  # x_0  # y_0
        )
    elif dim_z == 4:
        H = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=dt,  # x_0  # y_0
        )

    initial_location = np.array(
        [[10.0, 10.0, 0.0, 0.0]], dtype=dt
    ).T  # x, y, v_x, v_y

    initial_estimate_error = np.eye(dim_x, dtype=dt) * np.array(
        [1.0, 1.0, 2.0, 2.0], dtype=dt
    )

    measurement_noise = np.eye(dim_z, dtype=dt) * 0.01

    motion_noise = np.eye(dim_x, dtype=dt) * np.array(
        [10.0, 10.0, 10.0, 10.0], dtype=dt
    )

    # CPU
    start = time.time()
    for l in range(loops):

        if num_points == minimum:
            cpu_points = num_points
        else:
            cpu_points = 1

        for _ in range(cpu_points):
        # print("CPU at", l)

            np.random.seed(1234)

            f_fpy.x = initial_location

            # State space equation to estimate position and velocity
            f_fpy.F = F

            # only observable input is the x and y coordinates
            f_fpy.H = H

            # Covariance Matrix
            f_fpy.P = initial_estimate_error

            f_fpy.R = measurement_noise

            f_fpy.Q = motion_noise

        
            for _ in range(iterations):

                f_fpy.predict()

                z = np.random.random_sample(dim_z).astype(dt).T

                f_fpy.update(z)

    global cpu_baseline32
    global cpu_baseline64

    if num_points == minimum:
        if dt == np.float32:
            cpu_baseline32 = (time.time() - start) / loops
            cpu_time = cpu_baseline32
            # print("CPU Baseline:", cpu_baseline32)
        else:
            cpu_baseline64 = (time.time() - start) / loops
            cpu_time = cpu_baseline64
            # print("CPU Baseline:", cpu_baseline64)
    else:
        if dt == np.float32:
            cpu_time = (cpu_baseline32) * (num_points / minimum)
            # print("CPU:", cpu_time)
        else:
            cpu_time = (cpu_baseline64) * (num_points / minimum)
            # print("CPU:", cpu_time)

    print("CPU:", cpu_time)

    np.random.seed(1234)

    # GPU
    start = time.time()
    for l in range(loops):
        # print("GPU at", l)

        cuS.x = cp.repeat(
            cp.asarray(initial_location[cp.newaxis, :, :]), num_points, axis=0
        )

        # State space equation to estimate position and velocity
        cuS.F = cp.repeat(cp.asarray(F[cp.newaxis, :, :]), num_points, axis=0)

        # only observable input is the x and y coordinates
        cuS.H = cp.repeat(cp.asarray(H[cp.newaxis, :, :]), num_points, axis=0)

        # Covariance Matrix
        cuS.P = cp.repeat(
            cp.asarray(initial_estimate_error[cp.newaxis, :, :]),
            num_points,
            axis=0,
        )

        cuS.R = cp.repeat(
            cp.asarray(measurement_noise[cp.newaxis, :, :]), num_points, axis=0
        )

        cuS.Q = cp.repeat(
            cp.asarray(motion_noise[cp.newaxis, :, :]), num_points, axis=0
        )

        for _ in range(iterations):

            with prof.time_range("predict", 0):
                cuS.predict()

            z = np.atleast_2d(np.random.random_sample(dim_z).astype(dt)).T
            z = np.repeat(z[np.newaxis, :, :], num_points, axis=0)

            with cp.prof.time_range("update", 0):
                cuS.update(z)

            cp.cuda.runtime.deviceSynchronize()

    gpu_time = (time.time() - start) / loops
    print("GPU:", gpu_time)

    print("Speed Up", cpu_time / gpu_time)
    print()

    if debug is not None:
        print()
        print("Final")
        print("predicted filterpy")
        print(f_fpy.x)
        print("predicted cusignal (First)")
        print(cuS.x[0, :, :])
        print("predicted cusignal (Last)")
        print(cuS.x[-1, :, :])
        print()

    rtol = 1e-3

    if test is not None:
        np.testing.assert_allclose(f_fpy.x, cuS.x[0, :, :].get(), rtol)
        np.testing.assert_allclose(f_fpy.x, cuS.x[-1, :, :].get(), rtol)

    if debug is not None:
        print()
        print("Final")
        print("predicted filterpy")
        print(f_fpy.P)
        print("predicted cusignal (First)")
        print(cuS.P[0, :, :])
        print("predicted cusignal (Last)")
        print(cuS.P[-1, :, :])
        print()

    if test is not None:
        np.testing.assert_allclose(f_fpy.P, cuS.P[0, :, :].get(), rtol)
        np.testing.assert_allclose(f_fpy.P, cuS.P[-1, :, :].get(), rtol)


if __name__ == "__main__":
    num_points = [
        minimum,
        2 ** 14,
        2 ** 15,
        2 ** 16,
        2 ** 17,
        2 ** 18,
        2 ** 19,
        2 ** 20,
    ]

    dt = [np.float32, np.float64]

    for d, p in itertools.product(dt, num_points):
        main(d, p)
