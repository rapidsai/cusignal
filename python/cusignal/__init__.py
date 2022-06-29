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
    complex_cepstrum,
    inverse_complex_cepstrum,
    minimum_phase,
    real_cepstrum,
)
from cusignal.bsplines.bsplines import cubic, gauss_spline, quadratic
from cusignal.convolution.convolve import (
    choose_conv_method,
    convolve,
    convolve1d2o,
    convolve1d3o,
    convolve2d,
    fftconvolve,
)
from cusignal.convolution.correlate import correlate, correlate2d
from cusignal.demod.demod import fm_demod
try:
    from cusignal import diff
except:
   msg = """
   Warning - Could not find PyTorch. Please install to use
   differentiable functions in cuSignal.
   """
   print(msg)
from cusignal.estimation.filters import KalmanFilter
from cusignal.filter_design.fir_filter_design import (
    cmplx_sort,
    firwin,
    firwin2,
    kaiser_atten,
    kaiser_beta,
)
from cusignal.filtering.filtering import (
    channelize_poly,
    detrend,
    filtfilt,
    firfilter,
    firfilter2,
    firfilter_zi,
    freq_shift,
    hilbert,
    hilbert2,
    lfilter,
    lfilter_zi,
    sosfilt,
    wiener,
)
from cusignal.filtering.resample import decimate, resample, resample_poly, upfirdn
from cusignal.io.reader import read_bin, read_sigmf, unpack_bin
from cusignal.io.writer import pack_bin, write_bin, write_sigmf
from cusignal.peak_finding.peak_finding import argrelextrema, argrelmax, argrelmin
from cusignal.radartools.beamformers import mvdr
from cusignal.radartools.radartools import ambgfun, pulse_compression, pulse_doppler
from cusignal.spectral_analysis.spectral import (
    coherence,
    csd,
    istft,
    lombscargle,
    periodogram,
    spectrogram,
    stft,
    vectorstrength,
    welch,
)
from cusignal.utils.arraytools import (
    from_pycuda,
    get_pinned_array,
    get_pinned_mem,
    get_shared_array,
    get_shared_mem,
)
from cusignal.waveforms.waveforms import (
    chirp,
    gausspulse,
    sawtooth,
    square,
    unit_impulse,
)
from cusignal.wavelets.wavelets import cwt, morlet, morlet2, qmf, ricker
from cusignal.windows.windows import (
    barthann,
    bartlett,
    blackman,
    blackmanharris,
    bohman,
    boxcar,
    chebwin,
    cosine,
    exponential,
    flattop,
    gaussian,
    general_cosine,
    general_gaussian,
    general_hamming,
    get_window,
    hamming,
    hann,
    kaiser,
    nuttall,
    parzen,
    taylor,
    triang,
    tukey,
)

# Versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
