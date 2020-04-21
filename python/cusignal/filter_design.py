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


def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.
    The filter will have an angular (e.g. rad/s) cutoff frequency of 1.
    See Also
    --------
    butter : Filter design function using this prototype
    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = cp.array([])
    m = cp.arange(-N + 1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -cp.exp(1j * cp.pi * m / (2 * N))
    k = 1
    return z, p, k


filter_dict = {'butter': [buttap, buttord],
               'butterworth': [buttap, buttord],

               'cauer': [ellipap, ellipord],
               'elliptic': [ellipap, ellipord],
               'ellip': [ellipap, ellipord],

               'bessel': [besselap],
               'bessel_phase': [besselap],
               'bessel_delay': [besselap],
               'bessel_mag': [besselap],

               'cheby1': [cheb1ap, cheb1ord],
               'chebyshev1': [cheb1ap, cheb1ord],
               'chebyshevi': [cheb1ap, cheb1ord],

               'cheby2': [cheb2ap, cheb2ord],
               'chebyshev2': [cheb2ap, cheb2ord],
               'chebyshevii': [cheb2ap, cheb2ord],
               }

band_dict = {'band': 'bandpass',
             'bandpass': 'bandpass',
             'pass': 'bandpass',
             'bp': 'bandpass',

             'bs': 'bandstop',
             'bandstop': 'bandstop',
             'bands': 'bandstop',
             'stop': 'bandstop',

             'l': 'lowpass',
             'low': 'lowpass',
             'lowpass': 'lowpass',
             'lp': 'lowpass',

             'high': 'highpass',
             'highpass': 'highpass',
             'h': 'highpass',
             'hp': 'highpass',
             }

bessel_norms = {'bessel': 'phase',
                'bessel_phase': 'phase',
                'bessel_delay': 'delay',
                'bessel_mag': 'mag'}