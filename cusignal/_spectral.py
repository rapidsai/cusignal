import cupy as cp
from numba import cuda

# These cupy math functions aren't yet supported in numba
from math import sin, cos, atan2

@cuda.jit(fastmath=True)
def _lombscargle(x, y, freqs, pgram):
    """
    _lombscargle(x, y, freqs)
    Computes the Lomb-Scargle periodogram.
    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values (must be registered so the mean is zero).
    freqs : array_like
        Angular frequencies for output periodogram.
    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.
    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.
    See also
    --------
    lombscargle
    """
    
    F = cuda.grid(1)
    strideF = cuda.gridsize(1)
    
    for i in range(F, freqs.shape[0], strideF):
        
        freq = freqs[i]

        xc = 0.
        xs = 0.
        cc = 0.
        ss = 0.
        cs = 0.

        for j in range(x.shape[0]):

            c = cos(freq * x[j])
            s = sin(freq * x[j])

            xc += y[j] * c
            xs += y[j] * s
            cc += c * c
            ss += s * s
            cs += c * s

        tau = atan2(2.0 * cs, cc - ss) / (2.0 * freq)
        c_tau = cos(freq * tau)
        s_tau = sin(freq * tau)
        c_tau2 = c_tau * c_tau
        s_tau2 = s_tau * s_tau
        cs_tau = 2.0 * c_tau * s_tau

        pgram[i] = 0.5 * (((c_tau * xc + s_tau * xs)**2 / \
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
            ((c_tau * xs - s_tau * xc)**2 / \
            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)))


@cuda.jit(fastmath=True)
def _lombscargle_norm(x, y, freqs, pgram, y_dot):
    """
    _lombscargle(x, y, freqs)
    Computes the Lomb-Scargle periodogram.
    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values (must be registered so the mean is zero).
    freqs : array_like
        Angular frequencies for output periodogram.
    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.
    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.
    See also
    --------
    lombscargle
    """

    F = cuda.grid(1)
    strideF = cuda.gridsize(1)
    
    for i in range(F, freqs.shape[0], strideF):
        
        # Copy data to registers
        temp = 0.0
        freq = freqs[i]
        yD = 2.0 / y_dot[0]

        xc = 0.
        xs = 0.
        cc = 0.
        ss = 0.
        cs = 0.
        
        #cuda.syncthreads()

        for j in range(x.shape[0]):

            c = cos(freq * x[j])
            s = sin(freq * x[j])

            xc += y[j] * c
            xs += y[j] * s
            cc += c * c
            ss += s * s
            cs += c * s

        tau = atan2(2.0 * cs, cc - ss) / (2.0 * freq)
        c_tau = cos(freq * tau)
        s_tau = sin(freq * tau)
        c_tau2 = c_tau * c_tau
        s_tau2 = s_tau * s_tau
        cs_tau = 2.0 * c_tau * s_tau

        temp = 0.5 * (((c_tau * xc + s_tau * xs)**2 / \
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
            ((c_tau * xs - s_tau * xc)**2 / \
            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)))
        
        pgram[i] = temp * yD

""" 
import cupy as np
import cusignal
nin = 5
nout = 10
x = np.linspace(1, 10, nin)
y = np.sin(x)
f = np.linspace(1, 10, nout)
pgram = cusignal.lombscargle(x, y, f, normalize=True)
"""

""" 
for j in range(x.shape[0]):
    c = cos(freqs[i] * x[j])
    s = sin(freqs[i] * x[j])
    xc += y[j] * c
    xs += y[j] * s
    cc += c * c
    ss += s * s
    cs += c * s 
"""