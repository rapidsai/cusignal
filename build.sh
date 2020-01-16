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

VALIDARGS="clean cusignal -v -g -n --allgpuarch -h"
HELP="$0 [clean] [cusignal] [-v] [-g] [-n] [--allgpuarch] [-h]
   clean        - remove all existing build artifacts and configuration (start
                  over)
   cusignal     - build the cusignal Python package
   -v           - verbose build mode
   -g           - build for debug
   -n           - no install step
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

if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
    GPU_ARCH="-DGPU_ARCHS="
    echo "Building for the architecture of the GPU in the system..."
else
    GPU_ARCH="-DGPU_ARCHS=ALL"
    echo "Building for *ALL* supported GPU architectures..."
fi

################################################################################
# Build and install the cusignal Python package
if buildAll || hasArg cusignal; then

    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py install
    fi
fi
