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
    "rapids-build-env=$MINOR_VERSION.*"

# https://docs.rapids.ai/maintainers/depmgmt/ 
logger "Conda install custom..."
conda remove --force rapids-build-env rapids-notebook-env
conda install -c conda-forge scipy

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# logger "Build fatbins..."
# # Build fatbins
# SRC="cpp/src"
# FAT="python/cusignal"
# FLAGS="-std=c++11"

# if hasArg -p; then
#     FLAGS="${FLAGS} -Xptxas -v -Xptxas -warn-lmem-usage -Xptxas -warn-double-usage"
# fi

# GPU_ARCH="--generate-code arch=compute_35,code=sm_35 \
# --generate-code arch=compute_35,code=sm_37 \
# --generate-code arch=compute_50,code=sm_50 \
# --generate-code arch=compute_50,code=sm_52 \
# --generate-code arch=compute_53,code=sm_53 \
# --generate-code arch=compute_60,code=sm_60 \
# --generate-code arch=compute_61,code=sm_61 \
# --generate-code arch=compute_62,code=sm_62 \
# --generate-code arch=compute_70,code=sm_70 \
# --generate-code arch=compute_72,code=sm_72 \
# --generate-code arch=compute_75,code=[sm_75,compute_75]"

# echo "Building Convolution kernels..."
# FOLDER="convolution"
# mkdir -p ${FAT}/${FOLDER}/
# nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_convolution.cu -odir ${FAT}/${FOLDER}/ &

# echo "Building Filtering kernels..."
# FOLDER="filtering"
# mkdir -p ${FAT}/${FOLDER}/
# nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_upfirdn.cu -odir ${FAT}/${FOLDER}/ &
# nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_sosfilt.cu -odir ${FAT}/${FOLDER}/ &

# echo "Building IO kernels..."
# FOLDER="io"
# mkdir -p ${FAT}/${FOLDER}/
# nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_reader.cu -odir ${FAT}/${FOLDER}/ &
# nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_writer.cu -odir ${FAT}/${FOLDER}/ &

# echo "Building Spectral kernels..."
# FOLDER="spectral_analysis"
# mkdir -p ${FAT}/${FOLDER}/
# nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_spectral.cu -odir ${FAT}/${FOLDER}/ &

# wait

################################################################################

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
