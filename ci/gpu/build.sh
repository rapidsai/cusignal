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
    "rapids-build-env=$MINOR_VERSION.*" \
    "rapids-notebook-env=$MINOR_VERSION."

# https://docs.rapids.ai/maintainers/depmgmt/ 
# conda remove -f rapids-build-env rapids-notebook-env
# conda install "your-pkg=1.0.0"

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

conda remove -y --force blas nomkl rapids-build-env rapids-notebook-env
conda install -y -c pytorch "pytorch>=1.4"

${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log

