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

from cusignal.acoustics.cepstrum import rceps, cceps, cceps_unwrap
from cusignal.filtering.resample import (
    decimate,
    resample,
    resample_poly,
    upfirdn,
)
from cusignal.filtering.filtering import (
    wiener,
    lfiltic,
    sosfilt,
    hilbert,
    hilbert2,
    detrend,
    freq_shift,
)
from cusignal.convolution.correlate import correlate, correlate2d
from cusignal.convolution.convolve import (
    fftconvolve,
    choose_conv_method,
    convolve,
    convolve2d,
)
from cusignal.filter_design.fir_filter_design import (
    kaiser_beta,
    kaiser_atten,
    firwin,
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
    get_window,
)
from cusignal.spectral_analysis.spectral import (
    lombscargle,
    periodogram,
    welch,
    csd,
    spectrogram,
    stft,
    vectorstrength,
    coherence,
)
from cusignal.bsplines.bsplines import (
    gauss_spline,
    cubic,
    quadratic,
    cspline1d,
)
from cusignal.waveforms.waveforms import (
    square,
    gausspulse,
    chirp,
    unit_impulse,
)
from cusignal.wavelets.wavelets import qmf, morlet, ricker, cwt
from cusignal.peak_finding.peak_finding import (
    argrelmin,
    argrelmax,
    argrelextrema,
)
from cusignal.utils.arraytools import (
    get_shared_array,
    get_shared_mem,
    get_pinned_array
)
from cusignal.utils.compile_kernels import precompile_kernels
from cusignal.io.reader import (
    read_bin,
    write_bin,
    unpack_bin,
    pack_bin,
    read_sigmf,
    write_sigmf,
)

# Versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
