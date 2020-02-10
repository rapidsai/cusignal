#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##########################################
# cuSignal CPU conda build script for CI #
##########################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set versions of packages needed to be grabbed
export CUDF_VERSION=0.8.*
export NVSTRINGS_VERSION=0.8.*
export RMM_VERSION=0.8.*

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# BUILD - Conda package build
################################################################################

logger "Build conda pkg for cuSignal..."
conda build conda/recipes/cusignal --python=${PYTHON}

################################################################################
# UPLOAD - Conda packages
################################################################################

logger "Upload conda pkgs for cuSignal..."
source ci/cpu/upload-anaconda.sh
