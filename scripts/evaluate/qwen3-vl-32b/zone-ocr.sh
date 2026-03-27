#!/bin/bash

source .venv/bin/activate
set -euv

export HF_HOME=/tmp/hf-home
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_ALLREDUCE_USE_FLASHINFER=0

python3 -m evaluate.run --model qwen3-vl-32b --strategy zone-ocr
