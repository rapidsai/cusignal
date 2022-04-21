#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
#############################################
# cuSignal GPU build and test script for CI #
#############################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Parse git describei
cd "$WORKSPACE"
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
unset GIT_DESCRIBE_TAG

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD - Build cusignal
################################################################################

gpuci_logger "Build and install cusignal"
cd "${WORKSPACE}"
CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"
gpuci_conda_retry build --croot "${CONDA_BLD_DIR}" conda/recipes/cusignal --python=${PYTHON}
gpuci_mamba_retry install -c "${CONDA_BLD_DIR}" cusignal

################################################################################
# TEST - Run GoogleTest and py.tests for cusignal
################################################################################

set +e -Eo pipefail
EXITCODE=0
trap "EXITCODE=1" ERR

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
    exit 0
fi

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Python pytest for cusignal"
cd "$WORKSPACE/python"

pytest --cache-clear --junitxml="$WORKSPACE/junit-cusignal.xml" -v -s -m "not cpu"

"${WORKSPACE}/ci/gpu/test-notebooks.sh" 2>&1 | tee nbtest.log
python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log

return ${EXITCODE}
