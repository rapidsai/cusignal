# cuSignal 22.12.00 (Date TBD)

Please see https://github.com/rapidsai/cusignal/releases/tag/v22.12.00a for the latest changes to this development branch.

# cuSignal 22.10.00 (Date TBD)

Please see https://github.com/rapidsai/cusignal/releases/tag/v22.10.00a for the latest changes to this development branch.

# cuSignal 22.08.00 (Date TBD)

Please see https://github.com/rapidsai/cusignal/releases/tag/v22.08.00a for the latest changes to this development branch.

# cuSignal 22.06.00 (7 Jun 2022)

## 🛠️ Improvements

- Simplify conda recipes ([#484](https://github.com/rapidsai/cusignal/pull/484)) [@Ethyling](https://github.com/Ethyling)
- Use conda to build python packages during GPU tests ([#480](https://github.com/rapidsai/cusignal/pull/480)) [@Ethyling](https://github.com/Ethyling)
- Fix pinned buffer IO issues ([#479](https://github.com/rapidsai/cusignal/pull/479)) [@charlesbluca](https://github.com/charlesbluca)
- Use pre-commit to enforce Python style checks ([#478](https://github.com/rapidsai/cusignal/pull/478)) [@charlesbluca](https://github.com/charlesbluca)
- Extend `get_pinned_mem` to work with more dtype / shapes ([#477](https://github.com/rapidsai/cusignal/pull/477)) [@charlesbluca](https://github.com/charlesbluca)
- Add/fix installation sections for SDR example notebooks ([#476](https://github.com/rapidsai/cusignal/pull/476)) [@charlesbluca](https://github.com/charlesbluca)
- Use conda compilers ([#461](https://github.com/rapidsai/cusignal/pull/461)) [@Ethyling](https://github.com/Ethyling)
- Build packages using mambabuild ([#453](https://github.com/rapidsai/cusignal/pull/453)) [@Ethyling](https://github.com/Ethyling)

# cuSignal 22.04.00 (6 Apr 2022)

## 🐛 Bug Fixes

- Fix docs builds ([#455](https://github.com/rapidsai/cusignal/pull/455)) [@ajschmidt8](https://github.com/ajschmidt8)

## 📖 Documentation

- Fixes a list of minor errors in the example codes #457 ([#458](https://github.com/rapidsai/cusignal/pull/458)) [@sosae0](https://github.com/sosae0)

## 🚀 New Features

- adding complex parameter to chirp and additional tests ([#450](https://github.com/rapidsai/cusignal/pull/450)) [@mnicely](https://github.com/mnicely)

## 🛠️ Improvements

- Temporarily disable new `ops-bot` functionality ([#465](https://github.com/rapidsai/cusignal/pull/465)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add `.github/ops-bot.yaml` config file ([#463](https://github.com/rapidsai/cusignal/pull/463)) [@ajschmidt8](https://github.com/ajschmidt8)
- correlation lags function ([#459](https://github.com/rapidsai/cusignal/pull/459)) [@sosae0](https://github.com/sosae0)

# cuSignal 22.02.00 (2 Feb 2022)

## 🛠️ Improvements

- Allow CuPy 10 ([#448](https://github.com/rapidsai/cusignal/pull/448)) [@jakirkham](https://github.com/jakirkham)
- Speedup: Single-precision hilbert, resample, and lfilter_zi. ([#447](https://github.com/rapidsai/cusignal/pull/447)) [@luigifcruz](https://github.com/luigifcruz)
- Add Nemo Machine Translation to SDR Notebook ([#445](https://github.com/rapidsai/cusignal/pull/445)) [@awthomp](https://github.com/awthomp)
- Add citrinet and fm_demod cusignal function to notebook ([#444](https://github.com/rapidsai/cusignal/pull/444)) [@awthomp](https://github.com/awthomp)
- Add FM Demodulation to cuSignal ([#443](https://github.com/rapidsai/cusignal/pull/443)) [@awthomp](https://github.com/awthomp)
- Revamp Offline RTL-SDR Notebook - FM Demod and NeMo Speech to Text ([#442](https://github.com/rapidsai/cusignal/pull/442)) [@awthomp](https://github.com/awthomp)
- Bypass Covariance Matrix Calculation if Supplied in MVDR Beamformer ([#437](https://github.com/rapidsai/cusignal/pull/437)) [@awthomp](https://github.com/awthomp)

# cuSignal 21.12.00 (9 Dec 2021)

## 🐛 Bug Fixes

- Data type conversion for cwt. ([#429](https://github.com/rapidsai/cusignal/pull/429)) [@shevateng0](https://github.com/shevateng0)
- Fix indexing error in CWT ([#425](https://github.com/rapidsai/cusignal/pull/425)) [@awthomp](https://github.com/awthomp)

## 📖 Documentation

- Use PyData Sphinx Theme for Generated Documentation ([#436](https://github.com/rapidsai/cusignal/pull/436)) [@cmpadden](https://github.com/cmpadden)
- Doc fix for FIR Filters ([#426](https://github.com/rapidsai/cusignal/pull/426)) [@awthomp](https://github.com/awthomp)

## 🛠️ Improvements

- Fix Changelog Merge Conflicts for `branch-21.12` ([#439](https://github.com/rapidsai/cusignal/pull/439)) [@ajschmidt8](https://github.com/ajschmidt8)
- remove use_numba from notebooks - deprecated ([#433](https://github.com/rapidsai/cusignal/pull/433)) [@awthomp](https://github.com/awthomp)
- Allow complex wavelet output for morlet2 ([#428](https://github.com/rapidsai/cusignal/pull/428)) [@shevateng0](https://github.com/shevateng0)

# cuSignal 21.10.00 (7 Oct 2021)

## 🐛 Bug Fixes

- Change type check to use isinstance instead of str compare ([#415](https://github.com/rapidsai/cusignal/pull/415)) [@jrc-exp](https://github.com/jrc-exp)

## 📖 Documentation

- Fix typo in readme ([#413](https://github.com/rapidsai/cusignal/pull/413)) [@awthomp](https://github.com/awthomp)
- Add citation file ([#412](https://github.com/rapidsai/cusignal/pull/412)) [@awthomp](https://github.com/awthomp)
- README overhaul ([#411](https://github.com/rapidsai/cusignal/pull/411)) [@awthomp](https://github.com/awthomp)

## 🛠️ Improvements

- Fix Forward-Merge Conflicts ([#417](https://github.com/rapidsai/cusignal/pull/417)) [@ajschmidt8](https://github.com/ajschmidt8)
- Adding CFAR ([#409](https://github.com/rapidsai/cusignal/pull/409)) [@mbolding3](https://github.com/mbolding3)
- support space in workspace ([#349](https://github.com/rapidsai/cusignal/pull/349)) [@jolorunyomi](https://github.com/jolorunyomi)

# cuSignal 21.08.00 (4 Aug 2021)

## 🐛 Bug Fixes

- Remove pytorch from cusignal CI/CD ([#404](https://github.com/rapidsai/cusignal/pull/404)) [@awthomp](https://github.com/awthomp)
- fix firwin bug where fs is ignored if nyq provided ([#400](https://github.com/rapidsai/cusignal/pull/400)) [@awthomp](https://github.com/awthomp)
- Fixed imaginary part being removed in delay mode of ambgfun ([#397](https://github.com/rapidsai/cusignal/pull/397)) [@cliffburdick](https://github.com/cliffburdick)

## 🛠️ Improvements

- mvdr perf optimizations and addition of elementwise divide kernel ([#403](https://github.com/rapidsai/cusignal/pull/403)) [@awthomp](https://github.com/awthomp)
- Update sphinx config ([#395](https://github.com/rapidsai/cusignal/pull/395)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add Ambiguity Function (ambgfun) ([#393](https://github.com/rapidsai/cusignal/pull/393)) [@awthomp](https://github.com/awthomp)
- Fix `21.08` forward-merge conflicts ([#392](https://github.com/rapidsai/cusignal/pull/392)) [@ajschmidt8](https://github.com/ajschmidt8)
- Adding MVDR (Capon) Beamformer ([#383](https://github.com/rapidsai/cusignal/pull/383)) [@awthomp](https://github.com/awthomp)
- Fix merge conflicts ([#379](https://github.com/rapidsai/cusignal/pull/379)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuSignal 21.06.00 (9 Jun 2021)

## 🛠️ Improvements

- Perf Improvements for UPFIRDN ([#378](https://github.com/rapidsai/cusignal/pull/378)) [@mnicely](https://github.com/mnicely)
- Perf Improvements to SOS Filter ([#377](https://github.com/rapidsai/cusignal/pull/377)) [@mnicely](https://github.com/mnicely)
- Update environment variable used to determine `cuda_version` ([#376](https://github.com/rapidsai/cusignal/pull/376)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update `CHANGELOG.md` links for calver ([#373](https://github.com/rapidsai/cusignal/pull/373)) [@ajschmidt8](https://github.com/ajschmidt8)
- Merge `branch-0.19` into `branch-21.06` ([#372](https://github.com/rapidsai/cusignal/pull/372)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update docs build script ([#369](https://github.com/rapidsai/cusignal/pull/369)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuSignal 0.19.0 (21 Apr 2021)

## 🐛 Bug Fixes

- Fix bug in casting array to cupy ([#340](https://github.com/rapidsai/cusignal/pull/340)) [@awthomp](https://github.com/awthomp)

## 🚀 New Features

- Add morlet2 ([#336](https://github.com/rapidsai/cusignal/pull/336)) [@mnicely](https://github.com/mnicely)
- Increment Max CuPy Version in CI ([#328](https://github.com/rapidsai/cusignal/pull/328)) [@awthomp](https://github.com/awthomp)

## 🛠️ Improvements

- Add cusignal source dockerfile ([#343](https://github.com/rapidsai/cusignal/pull/343)) [@awthomp](https://github.com/awthomp)
- Update min scipy and cupy versions ([#339](https://github.com/rapidsai/cusignal/pull/339)) [@awthomp](https://github.com/awthomp)
- Add Taylor window ([#338](https://github.com/rapidsai/cusignal/pull/338)) [@mnicely](https://github.com/mnicely)
- Skip online signal processing tools testing ([#337](https://github.com/rapidsai/cusignal/pull/337)) [@awthomp](https://github.com/awthomp)
- Add 2D grid-stride loop to fix BUG 295 ([#335](https://github.com/rapidsai/cusignal/pull/335)) [@mnicely](https://github.com/mnicely)
- Update Changelog Link ([#334](https://github.com/rapidsai/cusignal/pull/334)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix bug in bin_reader that ignored dtype ([#333](https://github.com/rapidsai/cusignal/pull/333)) [@awthomp](https://github.com/awthomp)
- Remove six dependency ([#332](https://github.com/rapidsai/cusignal/pull/332)) [@awthomp](https://github.com/awthomp)
- Prepare Changelog for Automation ([#331](https://github.com/rapidsai/cusignal/pull/331)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update 0.18 changelog entry ([#330](https://github.com/rapidsai/cusignal/pull/330)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix merge conflicts in #315 ([#316](https://github.com/rapidsai/cusignal/pull/316)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuSignal 0.18.0 (24 Feb 2021)

## Bug Fixes 🐛

- Fix labeler.yml GitHub Action (#301) @ajschmidt8
- Fix Branch 0.18 merge 0.17 (#298) @BradReesWork

## Documentation 📖

- Add WSL instructions for cuSignal Windows Builds (#323) @awthomp
- Fix Radar API Docs (#311) @awthomp
- Update cuSignal Documentation to Include Radar Functions (#309) @awthomp
- Specify CuPy install time on Jetson Platform (#306) @awthomp
- Update README to optimize CuPy build time on Jetson (#305) @awthomp

## New Features 🚀

- Include scaffolding for new radar/phased array module and add new pulse compression feature (#300) @awthomp

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#321) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#319) @Ethyling
- Prepare Changelog for Automation (#314) @ajschmidt8
- Auto-label PRs based on their content (#313) @jolorunyomi
- Fix typo in convolution jupyter notebook example (#310) @awthomp
- Add Pulse-Doppler Processing to radartools (#307) @awthomp
- Create labeler.yml (#299) @jolorunyomi
- Clarify GPU timing in E2E Jupyter Notebook (#297) @awthomp
- Bump cuSignal Version (#296) @awthomp

# cuSignal 0.17.0 (10 Dec 2020)

## New Features
- PR #241 - Add inverse_complex_cepstrum and minimum_phase to acoustics module
- PR #270 - Add second and third order convolutions as convolve1d2o and convolve1d3o 
- PR #274 - Add nightly benchmarks to CI

## Improvements
- PR #267 - Various optimization across all functions
- PR #271 - Increase robustness of testing
- PR #280 - Fixing issue when reading sigmf data from parent folder
- PR #282 - Update README to reflect current versions

## Bug Fixes
- PR #272 Fix bug in gausspulse
- PR #275 Improve gpuCI Scripts
- PR #281 Fix erroneous path in CI scripts
- PR #286 Update conda install command in notebook

# cuSignal 0.16.0 (21 Oct 2020)

## New Features
- PR #185 - Add function to translate PyCUDA gpuarray to CuPy ndarray
- PR #195 - Add explicit FIR filter functionality
- PR #199 - Added Ampere support
- PR #208 - Remove CuPy v8 req for Kalman filter
- PR #210 - Add signal scope example to notebooks

## Improvements
- PR #196 - Update README for cuSignal 0.15
- PR #200 - Add if constexpr to binary reader
- PR #202 - Performance improvement to Lombscargle
- PR #203 - Update build process
- PR #207 - Add CUDA 10 compatibility with polyphase channelizer
- PR #211 - Add firfilter and channelize_poly to documentation; remove CPU only version of channelizer
- PR #212 - Add KF to documentation build
- PR #213 - Graceful handling of filter tap limit for polyphase channelizer
- PR #215 - Improve doc formating for filtering operations
- PR #219 - Add missing pytests
- PR #222 - Improved performance for various window functions
- PR #235 - Improve Wavelets functions performance
- PR #236 - Improve Bsplines functions performance
- PR #242 - Add PyTorch disclaimer to notebook
- PR #243 - Improve Peak Finding function performance
- PR #249 - Update README to add SDR integration instructions and improved install clarity
- PR #250 - Update ci/local/README.md
- PR #256 - Update to CuPy 8.0.0
- PR #260 - Optimize waveform functions

## Bug Fixes
- PR #214 - Fix grid-stride loop bug in polyphase channelizer
- PR #218 - Remove fatbins from source code on GH
- PR #221 - Synchronization issue with cusignal testing
- PR #237 - Update conda build files so fatbins are generated
- PR #239 - Fix mode issue in peak finding module
- PR #245 - Reduce number of default build architectures
- PR #246 - Remove lfiltic function
- PR #248 - Fix channelizer bugs
- PR #254 - Use CuPy v8 FFT cache plan
- PR #259 - Fix notebook error handling in gpuCI
- PR #263 - Remove precompile_kernels() from io_examples
- PR #264 - Fix Build error w/ nvidia-smi

# cuSignal 0.15.0 (26 Aug 2020)

## New Features
- PR #69 - Multi-point Kalman Filter
- PR #144 - Added AIR-T conda recipe
- PR #119 - Added code of conduct
- PR #122 - GPU accelerated SigMF Reader
- PR #130 - Reorg tests and benchmarks to match #83 code refactor
- PR #136 - Split reader and writer in IO packages and update docs
- PR #146 - Add compatibility with Scipy 1.5.0; Default to SciPy > 1.5
- PR #149 - Update Jetson conda to miniforge, Fix CoC, and Add SciPy Talk
- PR #148 - Load fatbin at runtime
- PR #69 - Multi-point Kalman Filter
- PR #158 - Add debug flag for debugging
- PR #161 - Basic implementation of polyphase channelizer

## Improvements
- PR #112 - Remove Numba kernels
- PR #121 - Add docs build script
- PR #126 - Install dependencies via meta package
- PR #132 - Add IO API guide Jupyter notebook
- PR #133 - Add IO module to cusignal docs
- PR #160 - Update KF functionality
- PR #170 - Combine tests and benchmarks
- PR #173 - Hardcode SOS width
- PR #181 - Added estimation notebook

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
- PR #155 - Update CI local docker build
- PR #153 - Fix issue with incorrect docker image being used in local build script
- PR #157 - Add docs for Kalman Filter
- PR #165 - Fix Kalman Filter version check
- PR #174 - Fix bug in KF script
- PR #175 - Update E2E Notebook for PyTorch 1.4, Fix SegFault
- PR #179 - Fix CuPy 8.0dev CI build error

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
- PR #101 - Add notebook testing to CI
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
