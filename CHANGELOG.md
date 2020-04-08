# cuSignal 0.14 (Date TBD)

## New Features
- PR #43 - Add pytest-benchmarks tests
- PR #48 - Addition of decimate for FIR ftypes
- PR #49 - Add CuPy Module for convolve2d and correlate2d

## Improvements
- PR #40 - Ability to specify time/freq domain for resample.
- PR #45 - Refactor `_signaltools.py` to use new Numba/CuPy framework
- PR #50 - Update README to reorganize install instructions

## Bug Fixes
- PR #44 - Fix issues in pytests 

# cuSignal 0.13 (31 Mar 2020)

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
- PR #26 - Cache raw CuPy kernels in upfirdn
- PR #28 - Add use_numba documentation in upfirdn and resample_poly; remove int support
- PR #29 - Fix typos in README
- PR #30 - Add Apache 2.0 license header to acoustics.py
- PR #31 - Adding stream support and additional data types to upfirdn
- PR #32 - Enable filter coefficient reuse across multiple calls to resample_poly and performance bug fixes
- PR #34 - Implement CuPy kernels as module object and templating
- PR #35 - Make upfirdn's kernel caching more generic; support 2D
- PR #36 - Set default upfirdn/resample_poly behavior to use Raw CuPy CUDA kernel rather than Numba; Doc updates

## Bug Fixes
- PR #4 - Direct method convolution isn't supported in CuPy, defaulting to NumPy [Examine in future for performance]
- PR #33 - Removes the conda test phase
- PR #37 - Fix docs to include all cuSignal modules
- PR #38 - Fix version in docs configuration

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
