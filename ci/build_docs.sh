#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml
  
rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
VERSION_NUMBER=$(rapids-get-rapids-version-from-git)

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  cusignal

# Build Python docs
gpuci_logger "Build Sphinx docs"
pushd docs
sphinx-build -b dirhtml source _html
sphinx-build -b text source _text
popd


if [[ ${RAPIDS_BUILD_TYPE} == "branch" ]]; then
  aws s3 sync --delete docs/_html "s3://rapidsai-docs/cusignal/${VERSION_NUMBER}/html"
  aws s3 sync --delete docs/_text "s3://rapidsai-docs/cusignal/${VERSION_NUMBER}/text"
fi