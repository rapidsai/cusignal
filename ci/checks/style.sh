#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#########################
# cuSignal Style Tester #
#########################

# Ignore errors and set path
set +e
PATH=/opt/conda/bin:$PATH

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run pre-commit hooks and get results/return code
pre-commit run --hook-stage manual --all-files
PRE_COMMIT_RETVAL=$?

RETVALS=(
  $PRE_COMMIT_RETVAL
)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
