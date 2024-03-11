#!/bin/bash

# set -e
CUDA_DEVICE=2
# Train & Evaluate
# Perform multiple repetitions of the same experiment, varying the split seed (for cross-validation)

TRANSLATORS=(
    mlp-1-layer
)

TOK_POSITIONS=(
    # tok_beginning_fixed_prompt
    tok_beginning_multi_prompts
    tok_in_place_multi_prompts
)

TRAINING_SETUPS=(
    with_entities
    with_entities_and_only_entities
)

FINETUNINGS=(
    # disabled
    shallow-vpt-5
    shallow-tpt-5
    shallow-vpt-5-tpt-5
)

IFS=,
TRANSLATORS="${TRANSLATORS[*]}"
TOK_POSITIONS="${TOK_POSITIONS[*]}"
TRAINING_SETUPS="${TRAINING_SETUPS[*]}"
FINETUNINGS="${FINETUNINGS[*]}"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n idclip \
    python train.py -m \
    finetuning="$FINETUNINGS" \
    training_setup="$TRAINING_SETUPS" \
    data/train="$TOK_POSITIONS" \
    model/translator="$TRANSLATORS"

