#!/usr/bin/env bash

export ROCM_ROOT=/opt/rocm-7.2.0
export LD_LIBRARY_PATH="$ROCM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ROCM_ROOT/bin${PATH:+:$PATH}"

