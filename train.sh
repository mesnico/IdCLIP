#!/bin/bash

set -e
CUDA_DEVICE=3
# Train & Evaluate
# Perform multiple repetitions of the same experiment, varying the split seed (for cross-validation)

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n idclip \
    python train.py -m \
    finetuning=shallow-vpt-5 \
    training_setup=with_entities_and_only_entities \
    data/train=tok_in_place,tok_beginning \

