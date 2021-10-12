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

_is_cupy_available = False

try:
    import cupy
    _is_cupy_available = True
except ImportError:
    pass

from cusignal.acoustics.cepstrum import (
    real_cepstrum,
    complex_cepstrum,
    inverse_complex_cepstrum,
    minimum_phase,
)
from cusignal.estimation.filters import KalmanFilter
from cusignal.filtering.resample import (
    decimate,
    resample,
    resample_poly,
    upfirdn,
)
from cusignal.filtering.filtering import (
    wiener,
    lfilter,
    lfilter_zi,
    firfilter,
    firfilter_zi,
    firfilter2,
    filtfilt,
    sosfilt,
    hilbert,
    hilbert2,
    detrend,
    channelize_poly,
    freq_shift,
)
from cusignal.convolution.correlate import correlate, correlate2d
from cusignal.convolution.convolve import (
    fftconvolve,
    choose_conv_method,
    convolve,
    convolve2d,
    convolve1d2o,
    convolve1d3o,
)
from cusignal.filter_design.fir_filter_design import (
    kaiser_beta,
    kaiser_atten,
    firwin,
    firwin2,
    cmplx_sort,
)
from cusignal.windows.windows import (
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
    taylor,
    get_window,
)
from cusignal.spectral_analysis.spectral import (
    lombscargle,
    periodogram,
    welch,
    csd,
    spectrogram,
    stft,
    istft,
    vectorstrength,
    coherence,
)
from cusignal.bsplines.bsplines import (
    gauss_spline,
    cubic,
    quadratic,
)
from cusignal.waveforms.waveforms import (
    sawtooth,
    square,
    gausspulse,
    chirp,
    unit_impulse,
)
from cusignal.wavelets.wavelets import qmf, morlet, ricker, morlet2, cwt
from cusignal.peak_finding.peak_finding import (
    argrelmin,
    argrelmax,
    argrelextrema,
)
from cusignal.utils.arraytools import (
    get_shared_array,
    get_shared_mem,
    get_pinned_array,
    get_pinned_mem,
    from_pycuda,
)
from cusignal.io.reader import (
    read_bin,
    unpack_bin,
    read_sigmf,
)
from cusignal.io.writer import (
    write_bin,
    pack_bin,
    write_sigmf,
)
from cusignal.radartools.radartools import (
    pulse_compression,
    pulse_doppler,
    ambgfun,
)
from cusignal.radartools.beamformers import (
    mvdr,
)

# Versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
