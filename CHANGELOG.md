# cuSignal 0.2 (Date TBD)

## Improvements
- PR #5 - Update cuSignal install directions for Windows OS.

## Bug Fixes
- PR #4 - Direct method convolution isn't supported in CuPy, defaulting to NumPy [Examine in future for performance]

# cuSignal 0.1 (04 Nov 2019)

## New Features

Initial commit of cuSignal featuring support for:

* **Convolution**
  * convolve - all methods other than direct
  * correlate - all methods other than direct
  * convolve2d
  * correlate2d
  * fftconvolve
  * choose_conv_method
* **B-splines**
  * cubic
  * quadratic
  * gauss_spline
* **Filtering**
  * wiener
  * lfiltic
  * hilbert
  * hilbert2
  * resample
  * resample_poly
  * upfirdn
* **Filter Design**
  * bilinear_zpk
  * firwin
  * iirfilter
  * cmplx_sort
* **Waveforms**
  * chirp
  * gausspulse
  * max_len_seq
  * square
* **Window Functions**
  * get_window
  * barthann
  * bartlett
  * blackman
  * blackmanharris
  * bohman
  * boxcar
  * chebwin
  * cosine
  * exponential
  * flattop
  * gaussian
  * general_cosine
  * general_gaussian
  * general_hamming
  * hamming
  * hann
  * kaiser
  * nuttall
  * triang
  * tukey
* **Wavelets**
  * morlet
  * ricker
  * cwt
* **Peak Finding**
  * argrelmin
  * argrelmax
  * rgrelextrema
* **Spectral Analysis**
  * periodogram
  * welch
  * csd
  * coherence
  * spectrogram
  * vectorstrength
  * stft
  * lombscargle
* **Extensions of Scipy Signal**
  * freq_shift - Frequency shift signal
