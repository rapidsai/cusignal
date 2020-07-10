# Test a kalman filter class

import cupy as cp
import numpy as np
import filterpy.kalman
import cusignal
import time
import itertools

dim_x = 4
dim_z = 3
loops = 1


def run_test(num_points, iterations, numba, dt):
    print("num_points", num_points)
    print("iterations", iterations)
    print("use_numba", numba)
    print("data type", dt)
    print("loops", loops)

    cuS = cusignal.KalmanFilter(
        num_points, dim_x, dim_z, dtype=dt, use_numba=numba
    )

    f_fpy = filterpy.kalman.KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # State Space Equations
    F = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],  # x = x0 + v_x*dt
            [0.0, 1.0, 0.0, 1.0],  # y = y0 + v_y*dt
            [0.0, 0.0, 1.0, 0.0],  # dx = v_x
            [1.0, 0.0, 0.0, 1.0],
            # [1.1, 0.2, -1.3, 99.4],  # x = x0 + v_x*dt
            # [0.1, -1.2, 99.3, 1.4],  # y = y0 + v_y*dt
            # [0.1, 99.2, -1.3, 0.4],  # dx = v_x
            # [1.1, -0.2, 99.3, -1.4],
        ],  # dy = v_y
        dtype=dt,
    )

    # Observability Input
    if dim_z == 2:
        H = np.array(
            [[1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0]], dtype=dt  # x_0  # y_0
        )
    elif dim_z == 3:
        H = np.array(
            [[1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0]], dtype=dt  # x_0  # y_0
        )
    elif dim_z == 4:
        H = np.array(
            [[1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0]], dtype=dt  # x_0  # y_0
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
        print("CPU at", l)

        f_fpy.x = initial_location

        # State space equation to estimate position and velocity
        f_fpy.F = F

        # only observable input is the x and y coordinates
        f_fpy.H = H

        # Covariance Matrix
        f_fpy.P = initial_estimate_error

        f_fpy.R = measurement_noise

        f_fpy.Q = motion_noise

        # z = np.asarray([0, 0, 0, 0], dtype=dt).T
        z = np.zeros(dim_z, dtype=dt).T

        for _ in range(1):
            for i in range(iterations):

                f_fpy.predict()

                # must be 2d for cuSignal.filter

                for j in range(dim_z):
                    z[j] = j

                f_fpy.update(z)

    print("CPU:", (time.time() - start) / loops)

    z = cp.zeros(dim_z, dtype=dt).T
    z = cp.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    # GPU
    start = time.time()
    for l in range(loops):
        print("GPU at", l)

        cuS.x = cp.repeat(
            cp.asarray(initial_location[cp.newaxis, :, :]), num_points, axis=0
        )

        # State space equation to estimate position and velocity
        cuS.F = cp.repeat(cp.asarray(F[cp.newaxis, :, :]), num_points, axis=0)

        # only observable input is the x and y coordinates
        cuS.H = cp.repeat(cp.asarray(H[cp.newaxis, :, :]), num_points, axis=0)

        # Covariance Matrix
        cuS.P = cp.repeat(
            cp.asarray(
                initial_estimate_error[cp.newaxis, :, :]
            ), num_points, axis=0
        )

        cuS.R = cp.repeat(
            cp.asarray(measurement_noise[cp.newaxis, :, :]), num_points, axis=0
        )

        cuS.Q = cp.repeat(
            cp.asarray(motion_noise[cp.newaxis, :, :]), num_points, axis=0
        )

        for i in range(iterations):

            cuS.predict()

            for j in range(dim_z):
                z[j] = j

            cuS.z = cp.repeat(z[cp.newaxis, :, :], num_points, axis=0)

            cuS.update()

            cp.cuda.runtime.deviceSynchronize()

    print("GPU:", (time.time() - start) / loops)

    # print()
    # print("Final")
    # print("predicted filterpy")
    # print(f_fpy.x)
    # print("predicted cusignal (First)")
    # print(cuS.x[0, :, :])
    # print("predicted cusignal (Last)")
    # print(cuS.x[-1, :, :])
    # print()

    rtol = 1e-5

    np.testing.assert_allclose(f_fpy.x, cuS.x[0, :, :].get(), rtol)
    np.testing.assert_allclose(f_fpy.x, cuS.x[-1, :, :].get(), rtol)

    # print()
    # print("Final")
    # print("predicted filterpy")
    # print(f_fpy.P)
    # print("predicted cusignal (First)")
    # print(cuS.P[0, :, :])
    # print("predicted cusignal (Last)")
    # print(cuS.P[-1, :, :])
    # print()

    np.testing.assert_allclose(f_fpy.P, cuS.P[0, :, :].get(), rtol)
    np.testing.assert_allclose(f_fpy.P, cuS.P[-1, :, :].get(), rtol)


# num_points = [2 ** 14, 2 ** 15, 2 ** 16, 2 ** 17, 2 ** 18, 2 ** 19, 2 ** 20]
num_points = [4096]
iterations = [1000]
numba = [False]
dt = [np.float64]

for p, i, n, d in itertools.product(num_points, iterations, numba, dt):
    run_test(p, i, n, d)
