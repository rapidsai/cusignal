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

    initial_location = np.array(
        [[10.0, 10.0, -99.0, -99.0]], dtype=dt
    ).T  # x, y, v_x, v_y

    # State Space Equations
    F = np.array(
        [
            [1.1, 0.2, -1.3, 99.4],  # x = x0 + v_x*dt
            [0.1, -1.2, 99.3, 1.4],  # y = y0 + v_y*dt
            [0.1, 99.2, -1.3, 0.4],  # dx = v_x
            [1.1, -0.2, 99.3, -1.4],
        ],  # dy = v_y
        dtype=dt,
    )

    # Observability Input
    H = np.array(
        [[1.0, -80.0, 1.0, -80.0], [-60.0, 1.0, -60.0, 1.0]], dtype=dt  # x_0  # y_0
    )

    initial_estimate_error = np.eye(dim_x, dtype=dt) * np.array(
        [1.9, -1.7, 2.6, -2.4], dtype=dt
    )
    measurement_noise = np.eye(dim_z, dtype=dt) * 0.01
    motion_noise = np.eye(dim_x, dtype=dt) * np.array(
        [10.1, -10.2, 10.3, -10.4], dtype=dt
    )

    f_fpy.x = initial_location
    print(initial_location.shape)
    cuS.x = cp.repeat(
        cp.asarray(initial_location[cp.newaxis, :, :]), num_points, axis=0
    )

    # State space equation to estimate position and velocity
    f_fpy.F = F
    cuS.F = cp.repeat(cp.asarray(F[cp.newaxis, :, :]), num_points, axis=0)

    # only observable input is the x and y coordinates
    f_fpy.H = H
    cuS.H = cp.repeat(cp.asarray(H[cp.newaxis, :, :]), num_points, axis=0)

    # Covariance Matrix
    f_fpy.P = initial_estimate_error
    cuS.P = cp.repeat(
        cp.asarray(
            initial_estimate_error[cp.newaxis, :, :]
        ), num_points, axis=0
    )

    f_fpy.R = measurement_noise
    cuS.R = cp.repeat(
        cp.asarray(measurement_noise[cp.newaxis, :, :]), num_points, axis=0
    )

    f_fpy.Q = motion_noise
    cuS.Q = cp.repeat(
        cp.asarray(motion_noise[cp.newaxis, :, :]), num_points, axis=0
    )

    # CPU
    start = time.time()
    for _ in range(loops):
        for _ in range(1):
            for i in range(iterations):

                f_fpy.predict()

                # must be 2d for cuSignal.filter
                z = np.array([i, i+1], dtype=dt).T

                print(f_fpy.x)

                # # print(z)
                # # print(z.T)
                # print(np.dot(f_fpy.H, f_fpy.x))
                # y = z - np.dot(f_fpy.H, f_fpy.x).T
                # # print(y.T)

                # temp = np.dot(f_fpy.H, np.dot(f_fpy.P, f_fpy.H.T)) + f_fpy.R
                # # print(temp)

                # K = np.dot(np.dot(f_fpy.P, f_fpy.H.T), np.linalg.inv(temp))
                # # print(K)

                # x = f_fpy.x * np.dot(K, y.T)
                # # print(x)

                # IKH = np.eye(dim_x) - np.dot(K, f_fpy.H)
                # # print(IKH)
                # # print(np.dot(K, f_fpy.H))

                # IKHP = np.dot(IKH, f_fpy.P)
                # # print(IKHP)

                # IKHPT = np.dot(IKHP, IKH.T)
                # print(IKHPT)

                # KR = np.dot(K, f_fpy.R)
                # print(KR)

                # KRKT = np.dot(KR, K.T)
                # print(KRKT)
                # print(IKHPT + KRKT)

                f_fpy.update(z)

    print("CPU:", (time.time() - start) / loops)

    z = cp.asarray([0, 0], dtype=dt).T  # must be 2d for cuSignal.filter
    z = cp.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    # GPU
    start = time.time()
    for _ in range(loops):
        for i in range(iterations):
            # print("iteration =", i)
            # print()

            cuS.predict()

            z[0] = i
            z[1] = i+1

            print(cuS.x[0, :, :])

            cuS.z = cp.repeat(z[cp.newaxis, :, :], num_points, axis=0)

            cuS.update()

            cp.cuda.runtime.deviceSynchronize()

    print("GPU:", (time.time() - start) / loops)

    print()
    print("Final")
    print("predicted filterpy")
    print(f_fpy.x)
    print("predicted cusignal (First)")
    print(cuS.x[0, :, :])
    # print("predicted cusignal (Last)")
    # print(cuS.x[-1, :, :])
    print()

    np.testing.assert_allclose(f_fpy.x, cuS.x[0, :, :].get(), 1e-6)
    np.testing.assert_allclose(f_fpy.x, cuS.x[-1, :, :].get(), 1e-6)

    print()
    print("Final")
    print("predicted filterpy")
    print(f_fpy.P)
    print("predicted cusignal (First)")
    print(cuS.P[0, :, :])
    # print("predicted cusignal (Last)")
    # print(cuS.P[-1, :, :])
    print()

    np.testing.assert_allclose(f_fpy.P, cuS.P[0, :, :].get(), 1e-6)
    np.testing.assert_allclose(f_fpy.P, cuS.P[-1, :, :].get(), 1e-6)


num_points = [2]
iterations = [4]
numba = [True]
dt = [np.float64]

for p, i, n, d in itertools.product(num_points, iterations, numba, dt):
    run_test(p, i, n, d)
