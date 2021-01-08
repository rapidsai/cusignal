# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
from ..windows.windows import get_window


def pulse_compression(x, template, normalize=False, window=None):
    """
    Pulse Compression is used to increase the range resolution and SNR
    by performing matched filtering of the transmitted pulse (template)
    with the received signal (x)

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    template : ndarray
        Transmitted signal, assume 1D array

    normalize : bool
        Normalize transmitted signal

     window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    Returns
    -------
    compressedIQ : ndarray
        Pulse compressed output
    """
    [num_pulses, samples_per_pulse] = x.shape

    if window is not None:
        Nx = len(template)
        if callable(window):
            W = window(cp.fft.fftfreq(Nx))
        elif isinstance(window, cp.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = get_window(window, Nx, False)

        template = cp.multiply(template, W)

    if normalize is True:
        template = cp.linalg.norm(template)

    fft_x = cp.fft.fft(x)
    fft_template = cp.conj(cp.tile(cp.fft.fft(template, samples_per_pulse),
                                   (num_pulses, 1)))
    compressedIQ = cp.fft.ifft(cp.multiply(fft_x, fft_template))

    return compressedIQ
