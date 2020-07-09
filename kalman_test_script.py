# Test a kalman filter class

import cupy as cp
import numpy as np
import filterpy.kalman
import cusignal
import time
import itertools

dim_x = 4
dim_z = 2
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

    f_fpy = filterpy.kalman.KalmanFilter(dim_x=4, dim_z=2)

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
    H = np.array(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=dt  # x_0  # y_0
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
    for _ in range(loops):

        f_fpy.x = initial_location

        # State space equation to estimate position and velocity
        f_fpy.F = F

        # only observable input is the x and y coordinates
        f_fpy.H = H

        # Covariance Matrix
        f_fpy.P = initial_estimate_error

        f_fpy.R = measurement_noise

        f_fpy.Q = motion_noise

        for _ in range(1):
            for i in range(iterations):

                f_fpy.predict()

                # must be 2d for cuSignal.filter
                z = np.array([i, i+1], dtype=dt).T

                f_fpy.update(z)

    print("CPU:", (time.time() - start) / loops)

    z = cp.asarray([0, 0], dtype=dt).T  # must be 2d for cuSignal.filter
    z = cp.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    # GPU
    start = time.time()
    for _ in range(loops):

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

            z[0] = i
            z[1] = i+1

            cuS.z = cp.repeat(z[cp.newaxis, :, :], num_points, axis=0)

            cuS.update()
            # print("GPU")
            # print("P", cuS.P[0, :, :])

            cp.cuda.runtime.deviceSynchronize()

    print("GPU:", (time.time() - start) / loops)

    print()
    print("Final")
    print("predicted filterpy")
    print(f_fpy.x)
    print("predicted cusignal (First)")
    print(cuS.x[0, :, :])
    # print("predicted cusignal (First)")
    # print(cuS.x[1, :, :])
    # print("predicted cusignal (Last)")
    # print(cuS.x[-2, :, :])
    print("predicted cusignal (Last)")
    print(cuS.x[-1, :, :])
    print()

    rtol = 1e-1

    # np.testing.assert_allclose(f_fpy.x, cuS.x[0, :, :].get(), rtol)
    # np.testing.assert_allclose(f_fpy.x, cuS.x[-1, :, :].get(), rtol)

    print()
    print("Final")
    print("predicted filterpy")
    print(f_fpy.P)
    print("predicted cusignal (First)")
    print(cuS.P[0, :, :])
    # print("predicted cusignal (First)")
    # print(cuS.P[1, :, :])
    # print("predicted cusignal (Last)")
    # print(cuS.P[-2, :, :])
    print("predicted cusignal (Last)")
    print(cuS.P[-1, :, :])
    print()

    # np.testing.assert_allclose(f_fpy.P, cuS.P[0, :, :].get(), rtol)
    # np.testing.assert_allclose(f_fpy.P, cuS.P[-1, :, :].get(), rtol)


num_points = [2**17]
iterations = [1]
numba = [True, False]
dt = [np.float64]

for p, i, n, d in itertools.product(num_points, iterations, numba, dt):
    run_test(p, i, n, d)
