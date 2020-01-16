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
    freq_shift
)
from cusignal.windows import (
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
from cusignal.bsplines import (
    gauss_spline,
    cubic,
    quadratic,
    cspline1d
)
from cusignal.waveforms import square, gausspulse, chirp, unit_impulse
from cusignal.wavelets import qmf, morlet, ricker, cwt
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
