# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
