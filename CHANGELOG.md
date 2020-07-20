
# cuSignal 0.15.0 (Date TBD)

## New Features
- PR #144 - Added AIR-T conda recipe
- PR #119 - Added code of conduct
- PR #122 - GPU accelerated SigMF Reader
- PR #130 - Reorg tests and benchmarks to match #83 code refactor
- PR #136 - Split reader and writer in IO packages and update docs
- PR #146 - Add compatibility with Scipy 1.5.0; Default to SciPy > 1.5
- PR #149 - Update Jetson conda to miniforge, Fix CoC, and Add SciPy Talk
- PR #69 - Multi-point Kalman Filter
- PR #158 - Add debug flag for debugging

## Improvements
- PR #112 - Remove Numba kernels
- PR #121 - Add docs build script
- PR #126 - Install dependencies via meta package
- PR #132 - Add IO API guide Jupyter notebook
- PR #133 - Add IO module to cusignal docs
- PR #160 - Update KF functionality

## Bug Fixes
- PR #164 - Fix typos in the rest of the example code.
- PR #162 - Fix typo in example plotting code
- PR #116 - Fix stream usage on CuPy raw kernels
- PR #124 - Remove cp_stream and autosync
- PR #127 - Fix selected documentation formatting errors
- PR #138 - Fix overflow issues in `upfirdn`
- PR #139 - Fixes packaging of python package
- PR #143 - Fix six package missing with Scipy 1.5
- PR #152 - Fix error in detrend related to missing indexing support with cp.r_
- PR #150 - Fix upfirdn output len for Scipy 1.5
- PR #153 - Fix issue with incorrect docker image being used in local build script
- PR #157 - Add docs for Kalman Filter

# cuSignal 0.14 (03 Jun 2020)

## New Features
- PR #43 - Add pytest-benchmarks tests
- PR #48 - Addition of decimate for FIR ftypes
- PR #49 - Add CuPy Module for convolve2d and correlate2d
- PR #51 - Add CuPy Module for lombscargle, along with tests/benchmarks
- PR #62 - Add CuPy Module for 1d convolve and correlate, along with tests/benchmarks
- PR #66 - Add CuPy Module for 2d upfirdn, along with tests/benchmarks
- PR #73 - Local gpuCI build script
- PR #75 - Add accelerated `lfilter` method.
- PR #82 - Implement `autosync` to synchronize raw kernels by default
- PR #99 - Implement `sosfilt` as an alternative to `lfilter`

## Improvements
- PR #40 - Ability to specify time/freq domain for resample.
- PR #45 - Refactor `_signaltools.py` to use new Numba/CuPy framework
- PR #50 - Update README to reorganize install instructions
- PR #55 - Update notebooks to use timeit instead of time
- PR #56 - Ability to precompile select Numba/CuPy kernel on import
- PR #60 - Updated decimate function to use an existing FIR window
- PR #61 - Fix link in README
- PR #65 - Added deprecation warning for Numba kernels
- PR #67 - cuSignal code refactor and documentation update
- PR #71 - README spelling and conda install fixes
- PR #78 - Ported lfilter to CuPy Raw Kernel (only 1D functional)
- PR #83 - Implement code refactor
- PR #84 - Update minimum versions of CuPy and Numba and associated conda envs
- PR #87 - Update lfilter documentation to clarifiy single-threaded perf
- PR #89 - Include data types in tests and benchmarks
- PR #95 - Add `.gitattributes` to remove notebooks from GitHub stats
- PR #97 - Add pytest-benchmark to conda ymls and update conda env name
- PR #98 - Update documentation to show pytest-benchmark usage and link to API docs
- PR #103 - Update notebooks to match new code structure
- PR #110 - Update README for 0.14 release
- PR #113 - Add git commit to conda package

## Bug Fixes
- PR #44 - Fix issues in pytests 
- PR #52 - Mirror API change in Numba 0.49
- PR #70 - Typo fix in docs api.rst that broke build
- PR #93 - Remove `lfilter` due to poor performance in real-time applications
- PR #96 - Move data type check to `_populate_kernel_cache`
- PR #104 - Fix flake8 errors

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
