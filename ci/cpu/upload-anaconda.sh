#!/bin/bash
set -e

CUDA_REL=${CUDA_VERSION%.*}

export UPLOADFILE=`gpuci_conda_retry build conda/recipes/cusignal --python=${PYTHON} --output`


LABEL_OPTION="--label main"
echo "LABEL_OPTION=${LABEL_OPTION}"

test -e ${UPLOADFILE}

# Restrict uploads to master branch
if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi

echo "Upload"
echo ${UPLOADFILE}
anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${UPLOADFILE}
