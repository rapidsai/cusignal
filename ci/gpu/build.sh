#!/bin/bash
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
#############################################
# cuSignal GPU build and test script for CI #
#############################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describei
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install -c rapidsai -c rapidsai-nightly -c nvidia -c conda-forge \
    cudatoolkit=${CUDA_REL} \
    "scipy>=1.3.0" \
    "numpy>=1.17.3" \
    boost \
    "numba>=0.49.0" \
    "cupy>=7.2.0" \
    pytest-benchmark \
    "ipython=7.3*" \
    jupyterlab \
    "torch>=1.4"
    matplotlib

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build cusignal
################################################################################

logger "Build cusignal..."
$WORKSPACE/build.sh clean cusignal

################################################################################
# TEST - Run GoogleTest and py.tests for cusignal
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
    exit 0
fi

logger "Check GPU usage..."
nvidia-smi

logger "Python pytest for cusignal..."
cd $WORKSPACE/python

pytest --cache-clear --junitxml=${WORKSPACE}/junit-cusignal.xml -v -s

${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log

