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

from cusignal.acoustics.cepstrum import (
    rceps,
    cceps,
    cceps_unwrap
)
from cusignal.signaltools import (
    correlate,
    fftconvolve,
    choose_conv_method,
    convolve,
    wiener,
    convolve2d,
    correlate2d,
    lfiltic,
    hilbert,
    hilbert2,
    cmplx_sort,
    resample,
    resample_poly,
    vectorstrength,
    detrend,
    freq_shift,
    decimate
)
from cusignal.window_functions.windows import (
    general_cosine,
    boxcar,
    triang,
    parzen,
    bohman,
    blackman,
    nuttall,
    blackmanharris,
    flattop,
    bartlett,
    hann,
    tukey,
    barthann,
    general_hamming,
    hamming,
    kaiser,
    gaussian,
    general_gaussian,
    chebwin,
    cosine,
    exponential,
    get_window
)
from cusignal.fir_filter_design import kaiser_beta, kaiser_atten, firwin
from cusignal.fftpack_helper import next_fast_len 
from cusignal.spectral import (
    lombscargle,
    periodogram,
    welch,
    csd,
    spectrogram,
    stft,
    coherence
)
from cusignal.bsplines.bsplines import (
    gauss_spline,
    cubic,
    quadratic,
    cspline1d
)
from cusignal.waveforms.waveforms import square, gausspulse, chirp, unit_impulse
from cusignal.wavelets.wavelets import qmf, morlet, ricker, cwt
from cusignal._peak_finding import argrelmin, argrelmax, argrelextrema
from cusignal._upfirdn import upfirdn
from cusignal._arraytools import (
    get_shared_array,
    get_shared_mem,
    axis_slice,
    axis_reverse,
    odd_ext,
    even_ext,
    const_ext,
    zero_ext,
    as_strided
)
from cusignal.cupy_helper import polyval, toeplitz, hankel

# Versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
