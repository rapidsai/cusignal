#!/bin/bash
set -e


if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

if [ -z "$TWINE_PASSWORD" ]; then
    echo "TWINE_PASSWORD not set"
    return 0
fi

echo "Upload pypi"
twine upload --skip-existing -u ${TWINE_USERNAME:-rapidsai} dist/*