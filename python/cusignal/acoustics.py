import cupy as cp
import cupyx.scipy.fftpack as fft
from math import floor, pi

def rceps(x, n=None, axis=-1):
    r"""
    Calculates the real cepstrum of an input sequence x where the cepstrum is
    defined as the inverse Fourier transform of the log magnitude DFT (spectrum)
    of a signal. It's primarily used for source/speaker separation in speech
    signal processing
    Parameters
    ----------
    x : ndarray
        Input sequence, if x is a matrix, return cepstrum in direction of axis
    n : int
        Size of Fourier Transform; If none, will use length of input array
    axis: int
        Direction for cepstrum calculation
    Returns
    -------
    ceps : ndarray
        Complex cepstrum result
    """
    x = cp.asarray(x)

    ceps = fft.ifft(cp.log(cp.abs(fft.fft(x, n=n, axis=axis))), n=n, axis=axis).real

    return ceps


def cceps_unwrap(x):
    r"""
    Unwrap phase for complex cepstrum calculation; helper function
    """
    x = cp.asarray(x)

    n = len(x)
    y = cp.unwrap(x)
    nh = floor((n+1)/2)
    nd = cp.round_(y[nh]/pi)
    y = y - cp.pi * nd * cp.arange(0, n)/nh

    return y


def cceps(x, n=None, axis=-1):
    r"""
    Calculates the complex cepstrum of a real valued input sequence x where the cepstrum is
    defined as the inverse Fourier transform of the log magnitude DFT (spectrum)
    of a signal. It's primarily used for source/speaker separation in speech
    signal processing.

    The input is altered to have zero-phase at pi radians (180 degrees)
    Parameters
    ----------
    x : ndarray
        Input sequence, if x is a matrix, return cepstrum in direction of axis
    n : int
       Size of Fourier Transform; If none, will use length of input array
    axis: int
        Direction for cepstrum calculation
    Returns
    -------
    ceps : ndarray
        Complex cepstrum result
    """
    x = cp.asarray(x)

    h = fft.fft(x, n=n, axis=axis)
    ah = cceps_unwrap(cp.angle(h))
    logh = cp.log(cp.abs(h)) + 1j*ah   
    cceps = fft.ifft(logh, n=n, axis=axis).real

    return cceps