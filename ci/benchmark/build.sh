#!/usr/bin/env bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
#############################################
# cuSignal's Benchmark test script for CI   #
#############################################

set -e
set -o pipefail
NUMARGS=$#
ARGS=$*

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set cleanup trap for Jenkins
if [ ! -z "$JENKINS_HOME" ] ; then
  gpuci_logger "Jenkins environment detected, setting cleanup trap"
  trap cleanup EXIT
fi

# Set path, build parallel level, and CUDA version
cd "$WORKSPACE"
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}

# Set home
export HOME="$WORKSPACE"

# Parse git describe
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Set Benchmark Vars
export BENCHMARKS_DIR="$WORKSPACE/benchmarks"

##########################################
# Environment Setup                      #
##########################################

# TODO: Delete build section when artifacts are available

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Install required packages"
gpuci_mamba_retry install -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia \
	"cudatoolkit=$CUDA_REL" \
	"rapids-build-env=${MINOR_VERSION}" \
	rapids-pytest-benchmark

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

##########################################
# Build cuSignal                         #
##########################################

gpuci_logger "Build cuSignal"
"$WORKSPACE/build.sh"

##########################################
# Run Benchmarks                         #
##########################################

BENCHMARK_META=$(jq -n \
  --arg NODE "${ASV_LABEL}" \
  --arg BRANCH "branch-${MINOR_VERSION}" '
  {
    "machineName": $NODE,
    "commitBranch": $BRANCH,
  }
')

echo "Benchmark meta:"
echo "${BENCHMARK_META}" | jq "."

gpuci_logger "Running Benchmarks"
cd $BENCHMARKS_DIR
pwd
set +e
time pytest ../python -k TestWavelets -v -m "not cpu" \
	--benchmark-enable \
	--benchmark-sort=mean \
	--benchmark-autosave \
	--benchmark-compare \
	--benchmark-compare-fail=mean:0.001 \
	--benchmark-gpu-device=0 \
    --benchmark-gpu-max-rounds=3 \
	--benchmark-asv-output-dir="${S3_ASV_DIR}" \
    --benchmark-asv-metadata="${BENCHMARK_META}"

EXITCODE=$?
