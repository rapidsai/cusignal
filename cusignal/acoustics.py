import cupy as cp
import cupyx.scipy.fftpack as fft

def rceps(x, nfft=None, axis=-1):
    r"""
    Calculates the real cepstrum of an input sequence x where the cepstrum is
    defined as the inverse Fourier transform of the log magnitude DFT (spectrum)
    of a signal. It's primarily used for source/speaker separation in speech
    signal processing
    Parameters
    ----------
    x : ndarray
        Input sequence, if x is a matrix, return cepstrum in direction of axis
    nfft : int
        Size of Fourier Transform
    axis: int
        Direction for cepstrum calculation
    Returns
    -------
    ceps : ndarray
        Complex cepstrum result
    """
    x = cp.asarray(x)

    if nfft is None:
        nfft = len(x)

    ceps = fft.ifft(cp.log(cp.abs(fft.fft(x, nfft))), nfft).real

    return ceps


def cceps_unwrap(x):
    r"""
    Unwrap phase for complex cepstrum calculation; helper function
    """
    x = cp.asarray(x)

    n = len(x)
    y = cp.unwrap(x)
    nh = cp.floor((n+1)/2)
    idx = nh + 1
    nd = cp.round(y[idx]/cp.pi)
    y = y - cp.pi * nd * cp.arange(0, n)/nh

    return y

def cceps(x, nfft=None, axis=-1):
    r"""
    Calculates the complex cepstrum of an input sequence x where the cepstrum is
    defined as the inverse Fourier transform of the log magnitude DFT (spectrum)
    of a signal. It's primarily used for source/speaker separation in speech
    signal processing.

    The input is altered to have zero-phase at pi radians (180 degrees)
    Parameters
    ----------
    x : ndarray
        Input sequence, if x is a matrix, return cepstrum in direction of axis
    nfft : int
        Size of Fourier Transform
    axis: int
        Direction for cepstrum calculation
    Returns
    -------
    ceps : ndarray
        Complex cepstrum result
    """
    x = cp.asarray(x)

    if nfft is None:
        nfft = len(x)

    h = fft.fft(x, nfft)
    ah = cceps_unwrap(cp.angle(h))
    logh = cp.log(cp.abs(h)) + 1j*ah   
    cceps = fft.ifft(logh).real

    return cceps