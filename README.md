# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuSignal</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cusignal/job/branches/job/cusignal-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cusignal/job/branches/job/cusignal-branch-pipeline/)

The [RAPIDS](https://rapids.ai) **cuSignal** project leverages [CuPy](https://github.com/cupy/cupy), [Numba](https://github.com/numba/numba), and the RAPIDS ecosystem for GPU accelerated signal processing. In some cases, cuSignal is a direct port of [Scipy Signal](https://github.com/scipy/scipy/tree/master/scipy/signal) to leverage GPU compute resources via CuPy but also contains Numba CUDA kernels for additional speedups for selected functions. cuSignal achieves its best gains on large signals and compute intensive functions but stresses online processing with zero-copy memory (pinned, mapped) between CPU and GPU.

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cusignal/blob/master/README.md) ensure you are on the `master` branch.

## Quick Start
cuSignal has an API that mimics SciPy Signal. In depth functionality is displayed in the [notebooks](https://github.com/rapidsai/cusignal/blob/master/notebooks) section of the repo, but let's examine the workflow for **Polyphase Resampling** under multiple scenarios:

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

gf = cusignal.resample_poly(gy, resample_up, resample_down, window=('kaiser', 0.5))
```
This code executes on an NVIDIA P100 in 258 ms.

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
gpu_signal = cusignal.get_shared_mem(num_samps, dtype=np.complex128)
gpu_signal[:] = cy

gf = cusignal.resample_poly(gpu_signal, resample_up, resample_down, window=('kaiser', 0.5))
```
This code executes on an NVIDIA P100 in 154 ms.

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

gf = cusignal.resample_poly(cp.asarray(cy), resample_up, resample_down, window=('kaiser', 0.5))
```
This code executes on an NVIDIA P100 in 728 ms.

## Dependencies
* NVIDIA GPU (Maxwell or Newer)
* CUDA Divers
* Anaconda/Miniconda (3.7 version)
* CuPy >= 6.2.0
* Optional: RTL-SDR or other SDR Driver/Packaging. Find more information and follow the instructions for setup [here](https://github.com/osmocom/rtl-sdr). NOTE: [pyrtlsdr](https://github.com/roger-/pyrtlsdr) is not automatically installed with the default cusignal environment. To make use of some of the examples in the Notebooks, you'll need to buy/install an rtl-sdr and necessary software packages.

## Install cuSignal, Linux OS, GeForce/Tesla/Quadro GPU with Anaconda

`conda install -c rapidsai-nightly -c conda-forge cusignal`

## Install cuSignal, Linux OS, GeForce/Tesla/Quadro GPU

1. Clone the repository

    ```bash
    # Set the localtion to cuSignal in an environment variable CUSIGNAL_HOME
    export CUSIGNAL_HOME=$(pwd)/cusignal

    # Download the cuSignal repo
    git clone https://github.com/rapidsai/cusignal.git $CUSIGNAL_HOME
    ```

2. Download and install [Andaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) then create the cuSignal conda environment:

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

    `conda activate cusignal`

4. Install cuSignal module

    ```bash
    cd $CUSIGNAL_HOME/python
    python setup.py install
    ```

    or

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

## Install cuSignal, Linux OS, Jetson Nano, Xavier, TX1, TX2

While there are many versions of Anaconda for AArch64 platforms, cuSignal has been tested and supports [conda4aarch64](https://github.com/jjhelmus/conda4aarch64/releases). Conda4aarch64 is also described in the [Numba aarch64 installation instructions](http://numba.pydata.org/numba-doc/latest/user/installing.html#installing-on-linux-armv8-aarch64-platforms). Further, it's assumed that your Jetson device is running a current edition of [JetPack](https://developer.nvidia.com/embedded/jetpack) and contains the CUDA Toolkit.

1. Clone the repository

    ```bash
    # Set the localtion to cuSignal in an environment variable CUSIGNAL_HOME
    export CUSIGNAL_HOME=$(pwd)/cusignal

    # Download the cuSignal repo
    git clone https://github.com/rapidsai/cusignal.git $CUSIGNAL_HOME
    ```

2. Install [conda4aarch64](https://github.com/jjhelmus/conda4aarch64/releases) and create the cuSignal conda environment:

    ```bash
    cd $CUSIGNAL_HOME
    conda env create -f conda/environments/cusignal_jetson_base.yml
    ```

3. Activate conda environment

    `conda activate cusignal`

4. Install cuSignal module

    ```bash
    cd $CUSIGNAL_HOME/python
    python setup.py install
    ```

    or

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

## Install cuSignal, Windows OS, GeForce/Tesla/Quadro GPU

1. Download and install [Andaconda](https://www.anaconda.com/distribution/) for Windows. In an Anaconda Prompt, navigate to your checkout of cuSignal.

2. Create cuSignal conda environment

    `conda create --name cusignal`

3. Activate conda environment

    `conda activate cusignal`

4. Install cuSignal Core Dependencies

    ```
    conda install numpy numba scipy cudatoolkit pip
    pip install cupy-cudaXXX
    ```

    Where XXX is the version of the CUDA toolkit you have installed. 10.1, for example is `cupy-cuda101`. See the [CuPy Documentation](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy) for information on getting Windows wheels for other versions of CUDA.

5. Install cuSignal module

    ```
    cd python
    python setup.py install
    ```

6. \[Optional\] Run tests
In the cuSignal top level directory:
    ```
    pip install pytest
    pytest
    ```

## Contributing Guide

Review the [CONTRIBUTING.md](https://github.com/rapidsai/cusignal/blob/master/CONTRIBUTING.md) file for information on how to contribute code and issues to the project.

## GTC DC Slides and Presentation
You can learn more about the cuSignal stack and motivations by viewing these GTC DC 2019 slides, located [here](https://drive.google.com/open?id=1rDNJVIHvCpFfNEDB9Gau5MzCN8G77lkH). The recording of this talk can be found at [GTC On Demand](https://on-demand.gputechconf.com/gtcdc/2019/video/dc91165-cusignal-gpu-acceleration-of-scipy-signal/)
