#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"
SUITEERROR=0

rapids-print-env

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  cusignal

rapids-logger "Check GPU usage"
nvidia-smi

set +e

# TODO: exit code handling is too verbose. Find a cleaner solution.
rapids-logger "pytest cusignal"
pushd python/cusignal
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cusignal.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=../.coveragerc \
  --cov=cusignal \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cusignal-coverage.xml" \
  --cov-report=term \
  test
exitcode=$?

if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in cusignal"
fi
popd

exit ${SUITEERROR}
