#!/bin/bash

#RAPIDS_DIR=/rapids
NOTEBOOKS_DIR="$WORKSPACE/notebooks"
NBTEST="$WORKSPACE/ci/utils/nbtest.sh"
LIBCUDF_KERNEL_CACHE_PATH="$WORKSPACE/.jitcache"

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)

SKIPNBS="sdr_wfm_demod.ipynb rtlsdr_offline_demod_to_transcript.ipynb io_examples.ipynb estimation_examples.ipynb E2E_Example.ipynb online_signal_processing_tools.ipynb" 

## Check env
env
conda info
conda config --show-sources
conda list --show-channel-urls

EXITCODE=0

for nb in $(find ${NOTEBOOKS_DIR} -name *.ipynb); do
    nbBasename=$(basename ${nb})

    # Skip all NBs that use dask (in the code or even in their name)
    if ((echo ${nb}|grep -qi dask) || \
        (grep -q dask ${nb})); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (suspected Dask usage, not currently automatable)"
        echo "--------------------------------------------------------------------------------"
    elif (echo " ${SKIPNBS} " | grep -q " ${nbBasename} "); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (listed in skip list)"
        echo "--------------------------------------------------------------------------------"
    else    
	# All notebooks are run from the directory in which they are contained.
        # This makes operations that assume relative paths easiest to understand
        # and maintain, since most users assume relative paths are relative to
        # the location of the notebook itself. After a run, the CWD must be
        # returned to NOTEBOOKS_DIR, since the find operation returned paths
        # relative to that dir.
        cd $(dirname ${nb})
        nvidia-smi
        ${NBTEST} ${nbBasename}
        EXITCODE=$((EXITCODE | $?))
        rm -rf ${LIBCUDF_KERNEL_CACHE_PATH}/*
        cd ${NOTEBOOKS_DIR}
    fi
done

nvidia-smi

exit ${EXITCODE}

