#!/bin/bash
# Copyright (c) 2020-2022, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate notebook testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  cusignal

NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"
pushd notebooks

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)
SKIPNBS="sdr_wfm_demod.ipynb io_examples.ipynb rtlsdr_offline_demod_to_transcript.ipynb"

# Set SUITEERROR to failure if any run fails
SUITEERROR=0

set +e
for nb in $(find . -name "*.ipynb"); do
    nbBasename=$(basename ${nb})
    # Skip all notebooks that use dask (in the code or even in their name)
    if ((echo ${nb} | grep -qi dask) || \
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
        pushd $(dirname ${nb})
        nvidia-smi
        ${NBTEST} ${nbBasename}
        SUITEERROR=$((SUITEERROR | $?))
        popd
    fi
done

exit ${SUITEERROR}
