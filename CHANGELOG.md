# cuSignal 0.13 (Date TBD)

## New Features
- PR #6 - Added Simple WFM Demodulation, Jupyter Notebook with SoapySDR integration
- PR #17 - Add conda recipe and gpuCI scripts
- PR #22 - Addition of real and complex cepstrum for speech/acoustic signal processing
- PR #25 - Raw CuPy Module for upfirdn

## Improvements
- PR #5 - Update cuSignal install directions for Windows OS.
- PR #9 - Update cuSignal documentation and Conda install ymls to better support Jetson Devices and prune dependencies
- PR #11 - Update cuSignal structure to match other RAPIDS projects
- PR #20 - Updated conda environment and README file

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
