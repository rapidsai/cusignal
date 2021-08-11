# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuSignal</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cusignal/job/branches/job/cusignal-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cusignal/job/branches/job/cusignal-branch-pipeline/)

The [RAPIDS](https://rapids.ai) **cuSignal** project leverages [CuPy](https://github.com/cupy/cupy), [Numba](https://github.com/numba/numba), and the RAPIDS ecosystem for GPU accelerated signal processing. In some cases, cuSignal is a direct port of [Scipy Signal](https://github.com/scipy/scipy/tree/master/scipy/signal) to leverage GPU compute resources via CuPy but also contains Numba CUDA and Raw CuPy CUDA kernels for additional speedups for selected functions. cuSignal achieves its best gains on large signals and compute intensive functions but stresses online processing with zero-copy memory (pinned, mapped) between CPU and GPU.

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cusignal/blob/main/README.md) ensure you are on the latest branch.

## Table of Contents
* [Quick Start](#quick-start)
* [Documentation](#documentation)
* [Installation](#installation)
    * [Conda: Linux OS](#conda-linux-os)
    * [Source: aarch64 (Jetson Nano, TK1, TX2, Xavier), Linux OS](#source-aarch64-jetson-nano-tk1-tx2-xavier-linux-os)
    * [Source: Linux OS](#source-linux-os)
    * [Source: Windows OS (with CUDA on WSL)](#source-windows-os)
    * [Docker](#docker---all-rapids-libraries-including-cusignal)
* [Software Defined Radio (SDR) Integration](#sdr-integration)
* [Benchmarking](#benchmarking)
* [Contribution Guide](#contributing-guide)
* [cuSignal Blogs and Talks](#cusignal-blogs-and-talks)


## Quick Start
cuSignal has an API that mimics SciPy Signal. In depth functionality is displayed in the [notebooks](https://github.com/rapidsai/cusignal/blob/main/notebooks) section of the repo, but let's examine the workflow for **Polyphase Resampling** under multiple scenarios:

**Scipy Signal (CPU)**
```python
import numpy as np
from scipy import signal

start = 0
stop = 10
num_samps = int(1e8)
resample_up = 2
resample_down = 3

cx = np.linspace(start, stop, num_samps, endpoint=False) 
cy = np.cos(-cx**2/6.0)

%%timeit
cf = signal.resample_poly(cy, resample_up, resample_down, window=('kaiser', 0.5))
```
This code executes on 2x Xeon E5-2600 in 2.36 sec.

**cuSignal with Data Generated on the GPU with CuPy**
```python
import cupy as cp
import cusignal

start = 0
stop = 10
num_samps = int(1e8)
resample_up = 2
resample_down = 3

gx = cp.linspace(start, stop, num_samps, endpoint=False) 
gy = cp.cos(-gx**2/6.0)

%%timeit
gf = cusignal.resample_poly(gy, resample_up, resample_down, window=('kaiser', 0.5))
```
This code executes on an NVIDIA V100 in 13.8 ms, a 170x increase over SciPy Signal

**cuSignal with Data Generated on the CPU with Mapped, Pinned (zero-copy) Memory**
```python
import cupy as cp
import numpy as np
import cusignal

start = 0
stop = 10
num_samps = int(1e8)
resample_up = 2
resample_down = 3

# Generate Data on CPU
cx = np.linspace(start, stop, num_samps, endpoint=False) 
cy = np.cos(-cx**2/6.0)

# Create shared memory between CPU and GPU and load with CPU signal (cy)
gpu_signal = cusignal.get_shared_mem(num_samps, dtype=np.float64)

%%time
# Move data to GPU/CPU shared buffer and run polyphase resampler
gpu_signal[:] = cy
gf = cusignal.resample_poly(gpu_signal, resample_up, resample_down, window=('kaiser', 0.5))
```
This code executes on an NVIDIA V100 in 174 ms.

**cuSignal with Data Generated on the CPU and Copied to GPU [AVOID THIS FOR ONLINE SIGNAL PROCESSING]**
```python
import cupy as cp
import numpy as np
import cusignal

start = 0
stop = 10
num_samps = int(1e8)
resample_up = 2
resample_down = 3

# Generate Data on CPU
cx = np.linspace(start, stop, num_samps, endpoint=False) 
cy = np.cos(-cx**2/6.0)

%%time
gf = cusignal.resample_poly(cp.asarray(cy), resample_up, resample_down, window=('kaiser', 0.5))
```
This code executes on an NVIDIA V100 in 637 ms.

## Documentation
The complete cuSignal API documentation including a complete list of functionality and examples can be found for both the Stable and Nightly (Experimental) releases.

[cuSignal 0.19 API](https://docs.rapids.ai/api/cusignal/stable/) | [cuSignal 0.20 Nightly](https://docs.rapids.ai/api/cusignal/nightly/)

### Installation
cuSignal has been tested on and supports all modern GPUs - from Maxwell to Ampere. While Anaconda is the preferred installation mechanism for cuSignal, developers and Jetson users should follow the source build instructions below. As of cuSignal 0.16, there isn't a cuSignal conda package for aarch64.

### Conda, Linux OS
cuSignal can be installed with conda ([Miniconda](https://docs.conda.io/en/latest/miniconda.html), or the full [Anaconda distribution](https://www.anaconda.com/distribution/)) from the `rapidsai` channel. If you're using a Jetson GPU, please follow the build instructions [below](https://github.com/rapidsai/cusignal#conda---jetson-nano-tk1-tx2-xavier-linux-os)

For `cusignal version == 0.19`:

```
For CUDA 10.1.2
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cusignal=0.19 python=3.8 cudatoolkit=10.1

# or, for CUDA 10.2
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cusignal=0.19 python=3.8 cudatoolkit=10.2

# or, for CUDA 11.0
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cusignal=0.19 python=3.8 cudatoolkit=11.0
```

For the nightly verison of `cusignal`, currently 0.20a:

```
# For CUDA 10.1.2
conda install -c rapidsai-nightly -c nvidia -c numba -c conda-forge \
    cusignal python=3.8 cudatoolkit=10.1.2

# or, for CUDA 10.2
conda install -c rapidsai-nightly -c nvidia -c numba -c conda-forge \
    cusignal python=3.8 cudatoolkit=10.2

# or, for CUDA 11.0
conda install -c rapidsai-nightly -c nvidia -c numba -c conda-forge \
    cusignal python=3.8 cudatoolkit=11.0
```

cuSignal has been tested and confirmed to work with Python 3.6, 3.7, and 3.8.

See the [Get RAPIDS version picker](https://rapids.ai/start.html) for more OS and version info.

### Source, aarch64 (Jetson Nano, TK1, TX2, Xavier), Linux OS

In cuSignal 0.15 and beyond, we are moving our supported aarch64 Anaconda environment from [conda4aarch64](https://github.com/jjhelmus/conda4aarch64/releases) to [miniforge](https://github.com/conda-forge/miniforge). Further, it's assumed that your Jetson device is running a current (>= 4.3) edition of [JetPack](https://developer.nvidia.com/embedded/jetpack) and contains the CUDA Toolkit.

1. Clone the repository

    ```bash
    # Set the location to cuSignal in an environment variable CUSIGNAL_HOME
    export CUSIGNAL_HOME=$(pwd)/cusignal

    # Download the cuSignal repo
    git clone https://github.com/rapidsai/cusignal.git $CUSIGNAL_HOME
    ```

2. Install [miniforge](https://github.com/conda-forge/miniforge) and create the cuSignal conda environment:

    ```bash
    cd $CUSIGNAL_HOME
    conda env create -f conda/environments/cusignal_jetson_base.yml
    ```

    Note: Compilation and installation of CuPy can be quite lengthy (~30+ mins), particularly on the Jetson Nano. Please consider setting this environment variable to decrease the CuPy dependency install time:

    `export CUPY_NVCC_GENERATE_CODE="arch=compute_XX,code=sm_XX"` with `XX` being your GPU's [compute capability](https://developer.nvidia.com/cuda-gpus#compute). If you'd like to compile to multiple architectures (e.g Nano and Xavier), concatenate the `arch=...` string with semicolins.

3. Activate conda environment

    `conda activate cusignal-dev`

4. Install cuSignal module

    ```bash
    cd $CUSIGNAL_HOME
    ./build.sh  # install cuSignal to $PREFIX if set, otherwise $CONDA_PREFIX
                # run ./build.sh -h to print the supported command line options.
    ```

5. Once installed, periodically update environment

    ```bash
    cd $CUSIGNAL_HOME
    conda env update -f conda/environments/cusignal_jetson_base.yml
    ```

6. Also, confirm unit testing via PyTest

    ```bash
    cd $CUSIGNAL_HOME/python
    pytest -v  # for verbose mode
    pytest -v -k <function name>  # for more select testing
    ```

### Source, Linux OS

1. Clone the repository

    ```bash
    # Set the location to cuSignal in an environment variable CUSIGNAL_HOME
    export CUSIGNAL_HOME=$(pwd)/cusignal

    # Download the cuSignal repo
    git clone https://github.com/rapidsai/cusignal.git $CUSIGNAL_HOME
    ```

2. Download and install [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) then create the cuSignal conda environment:

    **Base environment (core dependencies for cuSignal)**

    ```bash
    cd $CUSIGNAL_HOME
    conda env create -f conda/environments/cusignal_base.yml
    ```

    **Full environment (including RAPIDS's cuDF, cuML, cuGraph, and PyTorch)**

    ```bash
    cd $CUSIGNAL_HOME
    conda env create -f conda/environments/cusignal_full.yml
    ```

3. Activate conda environment

    `conda activate cusignal-dev`

4. Install cuSignal module

    ```bash
    cd $CUSIGNAL_HOME
    ./build.sh  # install cuSignal to $PREFIX if set, otherwise $CONDA_PREFIX
                # run ./build.sh -h to print the supported command line options.
    ```

5. Once installed, periodically update environment

    ```bash
    cd $CUSIGNAL_HOME
    conda env update -f conda/environments/cusignal_base.yml
    ```

6. Also, confirm unit testing via PyTest

    ```bash
    cd $CUSIGNAL_HOME/python
    pytest -v  # for verbose mode
    pytest -v -k <function name>  # for more select testing
    ```

### Source, Windows OS

We have confirmed that cuSignal successfully builds and runs on Windows by using [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Please follow the instructions in the link to install WSL 2 and the associated CUDA drivers. You can then proceed to follow the cuSignal source build instructions, below. 

1. Download and install [Andaconda](https://www.anaconda.com/distribution/) for Windows. In an Anaconda Prompt, navigate to your checkout of cuSignal.

2. Create cuSignal conda environment

    `conda create --name cusignal-dev`

3. Activate conda environment

    `conda activate cusignal-dev`

4. Install cuSignal Core Dependencies

    ```
    conda install numpy numba scipy cudatoolkit pip
    pip install cupy-cudaXXX
    ```

    Where XXX is the version of the CUDA toolkit you have installed. 10.1, for example is `cupy-cuda101`. See the [CuPy Documentation](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy) for information on getting Windows wheels for other versions of CUDA.

5. Install cuSignal module

    ```
    ./build.sh
    ```

6. \[Optional\] Run tests
In the cuSignal top level directory:
    ```
    pip install pytest
    pytest
    ```

### Docker - All RAPIDS Libraries, including cuSignal

cuSignal is part of the general RAPIDS docker container but can also be built using the included Dockerfile and the below instructions to build and run the container. Please note, `<image>` and `<tag>` are user specified, for example `docker build -t cusignal:cusignal-0.19 docker/.`.

```
docker build -t <image>:<tag> docker/.
docker run --gpus all --rm -it <image>:<tag> /bin/bash
```

To 
For `cusignal version == 0.19`:

```
# For CUDA 11.0
docker pull rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04
```

For the nightly version of `cusignal`
```
docker pull rapidsai/rapidsai-nightly:cuda11.0-runtime-ubuntu18.04
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai-nightly:cuda11.0-runtime-ubuntu18.04
```

Please see the [RAPIDS Release Selector](https://rapids.ai/start.html) for more information on supported Python, Linux, and CUDA versions.

## SDR Integration
[SoapySDR](https://github.com/pothosware/SoapySDR/wiki) is a "vendor neutral and platform independent" library for software-defined radio usage. When used in conjunction with device (SDR) specific modules, SoapySDR allows for easy command-and-control of radios from Python or C++. To install SoapySDR into an existing cuSignal Conda environment, run:

`conda install -c conda-forge soapysdr`

A full list of subsequent modules, specific to your SDR are listed [here](https://anaconda.org/search?q=soapysdr), but some common ones:
- rtlsdr: `conda install -c conda-forge soapysdr-module-rtlsdr`
- Pluto SDR: `conda install -c conda-forge soapysdr-module-plutosdr`
- UHD: `conda install -c conda-forge soapysdr-module-uhd`

Another popular SDR library, specific to the rtl-sdr, is [pyrtlsdr](https://github.com/roger-/pyrtlsdr).

For examples using SoapySDR, pyrtlsdr, and cuSignal, please see the [notebooks/sdr](https://github.com/rapidsai/cusignal/blob/main/notebooks/sdr) directory.

Please note, for most rtlsdr devices, you'll need to blacklist the libdvb driver in Linux. To do this, run `sudo vi /etc/modprobe.d/blacklist.conf` and add `blacklist dvb_usb_rtl28xxu` to the end of the file. Restart your computer upon completion.

If you have a SDR that isn't listed above (like the LimeSDR), don't worry! You can symbolically link the system-wide Python bindings installed via `apt-get` to the local conda environment. Please file an issue if you run into any problems.

## Benchmarking
cuSignal uses pytest-benchmark to compare performance between CPU and GPU signal processing implementations. To run cuSignal's benchmark suite, **navigate to the topmost python directory ($CUSIGNAL_HOME/python)** and run:

`pytest --benchmark-enable --benchmark-gpu-disable`

Benchmarks are disabled by default in `setup.cfg` providing only test correctness checks.

As with the standard pytest tool, the user can use the `-v` and `-k` flags for verbose mode and to select a specific benchmark to run. When intrepreting the output, we recommend comparing the _mean_ execution time reported.

To reduce columns in benchmark result's table, add `--benchmark-columns=LABELS`, like `--benchmark-columns=min,max,mean`.
For more information on `pytest-benchmark` please visit the [Usage Guide](https://pytest-benchmark.readthedocs.io/en/latest/usage.html).

Parameter `--benchmark-gpu-disable` is to disable memory checks from [Rapids GPU benchmark tool](https://github.com/rapidsai/benchmark). 
Doing so speeds up benchmarking.

If you wish to skip benchmarks of SciPy functions add `-m "not cpu"`

Lastly, benchmarks will be executed on local files. Therefore to test recent changes made to source, rebuild cuSignal.

### Example
`pytest -k upfirdn2d -m "not cpu" --benchmark-enable --benchmark-gpu-disable --benchmark-columns=mean`

### Output
```bash
cusignal/test/test_filtering.py ..................                                                                                                                                                                                                                                   [100%]


---------- benchmark 'UpFirDn2d': 18 tests -----------
Name (time in us, mem in bytes)         Mean          
------------------------------------------------------
test_upfirdn2d_gpu[-1-1-3-256]      195.2299 (1.0)    
test_upfirdn2d_gpu[-1-9-3-256]      196.1766 (1.00)   
test_upfirdn2d_gpu[-1-1-7-256]      196.2881 (1.01)   
test_upfirdn2d_gpu[0-2-3-256]       196.9984 (1.01)   
test_upfirdn2d_gpu[0-9-3-256]       197.5675 (1.01)   
test_upfirdn2d_gpu[0-1-7-256]       197.9015 (1.01)   
test_upfirdn2d_gpu[-1-9-7-256]      198.0923 (1.01)   
test_upfirdn2d_gpu[-1-2-7-256]      198.3325 (1.02)   
test_upfirdn2d_gpu[0-2-7-256]       198.4676 (1.02)   
test_upfirdn2d_gpu[0-9-7-256]       198.6437 (1.02)   
test_upfirdn2d_gpu[0-1-3-256]       198.7477 (1.02)   
test_upfirdn2d_gpu[-1-2-3-256]      200.1589 (1.03)   
test_upfirdn2d_gpu[-1-2-2-256]      213.0316 (1.09)   
test_upfirdn2d_gpu[0-1-2-256]       213.0944 (1.09)   
test_upfirdn2d_gpu[-1-9-2-256]      214.6168 (1.10)   
test_upfirdn2d_gpu[0-2-2-256]       214.6975 (1.10)   
test_upfirdn2d_gpu[-1-1-2-256]      216.4033 (1.11)   
test_upfirdn2d_gpu[0-9-2-256]       217.1675 (1.11)   
------------------------------------------------------
```

## Contributing Guide

Review the [CONTRIBUTING.md](https://github.com/rapidsai/cusignal/blob/main/CONTRIBUTING.md) file for information on how to contribute code and issues to the project.

## cuSignal Blogs and Talks
* cuSignal - GPU Accelerating SciPy Signal with Numba and CuPy cuSignal - SciPy 2020 - [Recording](https://youtu.be/yYlX2bbdXDk)
* Announcement Talk - GTC DC 2019 - [Recording](https://on-demand.gputechconf.com/gtcdc/2019/video/dc91165-cusignal-gpu-acceleration-of-scipy-signal/) | [Slides](https://on-demand.gputechconf.com/gtcdc/2019/pdf/dc91165-cusignal-gpu-acceleration-of-scipy-signal.pdf)
* [GPU Accelerated Signal Processing with cuSignal](https://medium.com/rapids-ai/gpu-accelerated-signal-processing-with-cusignal-689062a6af8) - Adam Thompson - Medium
* [cuSignal 0.13 - Entering the Big Leagues and Focused on Screamin' Streaming Performance](https://medium.com/rapids-ai/cusignal-0-13-entering-the-big-leagues-and-focused-on-screamin-streaming-performance-141908b10b3b) - Adam Thompson - Medium
* [cuSignal: Easy CUDA GPU Acceleration for SDR DSP and Other Applications](https://www.rtl-sdr.com/cusignal-easy-cuda-gpu-acceleration-for-sdr-dsp-and-other-applications/) - RTL-SDR.com
* [cuSignal on the AIR-T](http://docs.deepwavedigital.com/Tutorials/7_cuSignal.html) - Deepwave Digital
* [Detecting, Labeling, and Recording Training Data with the AIR-T and cuSignal](https://www.youtube.com/watch?v=yhVm9hH4nIo) - Deepwave Digital
* [Signal Processing and Deep Learning](https://www.youtube.com/watch?v=S17vUaTDHts) - Deepwave Digital
* [cuSignal and CyberRadio Demonstrate GPU Accelerated SDR](https://limemicro.com/news/cusignal-and-cyberradio-demonstrate-gpu-accelerated-sdr/) - Andrew Back - LimeMicro
* [Follow the latest cuSignal Announcements on Twitter](https://twitter.com/hashtag/cusignal?f=live)
