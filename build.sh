#!/bin/bash

# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cuSignal build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean cusignal -c -v -g -n -p --allgpuarch -h"
HELP="$0 [clean] [cusignal] [-v] [-g] [-n] [--allgpuarch] [-h]
   clean        - remove all existing build artifacts and configuration (start
                  over)
   cusignal     - build the cusignal Python package
   -c           - ci build
   -v           - verbose build mode
   -g           - build for debug
   -n           - no install step
   -p           - Pass additional Xptxas options
   --allgpuarch - build for all supported GPU architectures
   -h           - print this text

   default action (no args) is to build and install 'cusignal' target
"
CUSIGNAL_BUILD_DIR=${REPODIR}/python/build
BUILD_DIRS="${CUSIGNAL_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_ALL_GPU_ARCH=0

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function buildAll {
    ((${NUMARGS} == 0 )) || !(echo " ${ARGS} " | grep -q " [^-]\+ ")
}

if hasArg -h; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE=1
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
    if [ -d ${bd} ]; then
        find ${bd} -mindepth 1 -delete
        rmdir ${bd} || true
    fi
    done
fi

GET_CC(){
    MAJOR=`python3 -c 'from ctypes import *; \
        sms = c_int(); \
        l = cdll.LoadLibrary("libcuda.so"); \
        l.cuInit(0); \
        l.cuDeviceGetAttribute(byref(sms), 75, '${1}'); \
        print(sms.value)'`
    MINOR=`python3 -c 'from ctypes import *; \
        sms = c_int(); \
        l = cdll.LoadLibrary("libcuda.so"); \
        l.cuInit(0); \
        l.cuDeviceGetAttribute(byref(sms), 76, '${1}'); \
        print(sms.value)'`
}

if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
    echo "Building for the architecture of the GPU in the system..."
    DEVICES=${CUDA_VISIBLE_DEVICES}
    if [ -z "${DEVICES}" ]
    then
        # If DEVICES is empty, retrieve all attached NVIDIA GPU
        NUMGPU=`nvidia-smi -L | wc -l`
        for (( i=0; i<${NUMGPU}; i++ ))
        do
            GET_CC ${i}
            GPU_ARCH="${GPU_ARCH} --generate-code arch=compute_${MAJOR}${MINOR},code=sm_${MAJOR}${MINOR}"
        done
    else
        IFS=',' read -r -a arr <<< ${DEVICES}
        for i in ${arr[@]}
        do
            GET_CC ${i}
            GPU_ARCH="${GPU_ARCH} --generate-code arch=compute_${MAJOR}${MINOR},code=sm_${MAJOR}${MINOR}"
        done
    fi
    
else
    echo "Building for *ALL* supported GPU architectures..."

    GPU_ARCH="--generate-code arch=compute_60,code=sm_60 \
    --generate-code arch=compute_61,code=sm_61 \
    --generate-code arch=compute_62,code=sm_62 \
    --generate-code arch=compute_70,code=sm_70 \
    --generate-code arch=compute_72,code=sm_72"

    if [ "$NVCC_V" -lt 11 ]; then
        GPU_ARCH="${GPU_ARCH} --generate-code arch=compute_75,code=[sm_75,compute_75]"
    else
        GPU_ARCH="${GPU_ARCH} --generate-code arch=compute_75,code=sm_75 \
        --generate-code arch=compute_80,code=[sm_80,compute_80]"
    fi
fi

################################################################################
# Build fatbins
SRC="cpp/src"
FAT="python/cusignal"
NVCC_V=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -f1 -d'.')
GCC_V=$(gcc --version | grep gcc | cut -f2 -d')' | cut -f1 -d'.' | xargs)

# Must check GCC for Centos OS
if [ "$GCC_V" -lt 7 ] || [ "$NVCC_V" -lt 11 ]; then
    FLAGS="-std=c++11"
else
    FLAGS="-std=c++17"
fi

if hasArg -v; then
    FLAGS="${FLAGS} -Xptxas -v"
fi

if hasArg -p; then
    FLAGS="${FLAGS} -Xptxas -warn-lmem-usage -Xptxas -warn-double-usage"
fi

echo "Building Convolution kernels..."
FOLDER="convolution"
mkdir -p ${FAT}/${FOLDER}/
nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_convolution.cu -odir ${FAT}/${FOLDER}/ &

echo "Building Filtering kernels..."
FOLDER="filtering"
mkdir -p ${FAT}/${FOLDER}/
nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_upfirdn.cu -odir ${FAT}/${FOLDER}/ &
nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_sosfilt.cu -odir ${FAT}/${FOLDER}/ &
nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_channelizer.cu -odir ${FAT}/${FOLDER}/ &

echo "Building IO kernels..."
FOLDER="io"
mkdir -p ${FAT}/${FOLDER}/
nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_reader.cu -odir ${FAT}/${FOLDER}/ &
nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_writer.cu -odir ${FAT}/${FOLDER}/ &

echo "Building Peak Finding kernels..."
FOLDER="peak_finding"
mkdir -p ${FAT}/${FOLDER}/
nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_peak_finding.cu -odir ${FAT}/${FOLDER}/ &

echo "Building Spectral kernels..."
FOLDER="spectral_analysis"
mkdir -p ${FAT}/${FOLDER}/
nvcc --fatbin ${FLAGS} ${GPU_ARCH} ${SRC}/${FOLDER}/_spectral.cu -odir ${FAT}/${FOLDER}/ &

wait

################################################################################
# Build and install the cusignal Python package
if buildAll || hasArg cusignal; then

    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
        if hasArg -c; then
            python setup.py install --single-version-externally-managed --record record.txt
        else
            python setup.py install
        fi
    fi
fi
