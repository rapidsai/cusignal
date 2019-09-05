# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuSignal</div>

The [RAPIDS](https://rapids.ai) **cuSignal** project leverages [CuPy](https://github.com/cupy/cupy), [Numba](https://github.com/numba/numba), and the RAPIDS ecosystem for GPU accelerated signal processing. In some cases, cuSignal is a direct port of [Scipy Signal](https://github.com/scipy/scipy/tree/master/scipy/signal) to leverage GPU compute resources via CuPy but also contains Numba CUDA kernels for additional speedups for selected functions. cuSignal achieves its best gains on large signals and compute intensive functions but stresses online processing with zero-copy memory (pinned, mapped) between CPU and GPU.

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cusignal/blob/master/README.md) ensure you are on the `master` branch.

## Quick Start

## Dependencies
* NVIDIA GPU (Pascal or Newer)
* CUDA Divers
* Anaconda/Miniconda (3.7 version)
* CuPy >= 7.0.0
* Optional: RTL-SDR or other SDR Driver/Packaging. Find more information and follow the instructions for setup [here](https://github.com/osmocom/rtl-sdr). NOTE: [pyrtlsdr](https://github.com/roger-/pyrtlsdr) is automatically installed with the default cusignal environment. To make use of some of the examples in the Notebooks, you'll need to buy/install an rtl-sdr.

## Install cuSignal

### Download and install Andaconda then create conda environment. 
`conda env create -f cusignal_conda_env.yml`

### Activate conda environment

`conda activate cusignal`

### Install cuSignal module

`python setup.py install`

### Once installed, periodically update environment

`conda env update -f cusignal_conda_env.yml`

## Contributing Guide

Review the [CONTRIBUTING.md](https://github.com/rapidsai/cusignal/blob/master/CONTRIBUTING.md) file for information on how to contribute code and issues to the project.
