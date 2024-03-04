#!/bin/bash

CUDA_DEVICE=3
EXP_ROOT="./runs"

CKPT=best-entities-sum #best-contrastive-sum # best-entities-sum

DATA=coco_faceswap_5_entities
MODEL=idclip
LOSS=info-nce
LR="5e-05"

TEST_CONFIGS=(
    entities_retrieval
    general_retrieval_tok_in_place
    general_retrieval_tok_beginning
)

TRANSLATORS=(
    mlp-1-layer
)

TOK_POSITIONS=(
    tok_in_place
    tok_beginning
)

TRAINING_SETUPS=(
    with_entities
    with_entities_and_only_entities
)

FINETUNINGS=(
    disabled
    shallow-vpt-5
)

for TRANSLATOR in ${TRANSLATORS[@]}; do
    for TOK_POSITION in ${TOK_POSITIONS[@]}; do
        for TRAINING_SETUP in ${TRAINING_SETUPS[@]}; do
            for FINETUNING in ${FINETUNINGS[@]}; do
                
                EXP_PATH=${EXP_ROOT}/data=${DATA}/model=${MODEL}/translator=${TRANSLATOR}/loss=${LOSS}/tok_position=${TOK_POSITION}/training-setup=${TRAINING_SETUP}/finetuning=${FINETUNING}/lr=${LR}
                echo "Evaluating $EXP_PATH"

                    # evaluate the various test protocols
                    for TEST_CONFIG in ${TEST_CONFIGS[@]}; do
                        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n idclip python retrieval.py run_dir="\"$EXP_PATH\"" ckpt=${CKPT} data/test=${TEST_CONFIG}
                    done
                done
            done
        done
    done
done