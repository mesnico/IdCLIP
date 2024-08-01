#!/bin/bash

CUDA_DEVICE=2
EXP_ROOT="./runs"

DATA=coco_faceswap_5_entities
MODEL=idclip
LR="5e-05"

CKPTS=(
    best-entities-sum 
    best-contrastive-sum
)

TEST_CONFIGS=(
    entities_retrieval_conf1
    entities_retrieval_conf2
    entities_retrieval_conf3
    entities_retrieval_conf4
    entities_retrieval_conf5
    general_retrieval_conf1
    general_retrieval_conf2
    general_retrieval_conf3
    general_retrieval_conf4
    general_retrieval_conf5
)

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
    # with_entities_and_only_entities_weighted_0.10
    # with_entities_and_only_entities_weighted_0.02
    # with_entities_and_only_entities_weighted_1.00
    # only_entities
)

FINETUNINGS=(
    disabled
    shallow-vpt-5
    # shallow-tpt-5
    # shallow-vpt-5-tpt-5
)

LOSSES=(
    info-nce
    # info-nce-entity-bce
)

SUCCESS=0
TOTAL=0

for CKPT in ${CKPTS[@]}; do
    for TRANSLATOR in ${TRANSLATORS[@]}; do
        for LOSS in ${LOSSES[@]}; do
            for TOK_POSITION in ${TOK_POSITIONS[@]}; do
                for TRAINING_SETUP in ${TRAINING_SETUPS[@]}; do
                    for FINETUNING in ${FINETUNINGS[@]}; do
                    
                        EXP_PATH=${EXP_ROOT}/data=${DATA}/model=${MODEL}/translator=${TRANSLATOR}/loss=${LOSS}/tok_position=${TOK_POSITION}/training-setup=${TRAINING_SETUP}/finetuning=${FINETUNING}/lr=${LR}
                        echo "Evaluating $EXP_PATH"

                        # evaluate the various test protocols
                        for TEST_CONFIG in ${TEST_CONFIGS[@]}; do
                            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n idclip python retrieval.py run_dir="\"$EXP_PATH\"" ckpt=${CKPT} data/test=${TEST_CONFIG}
                            if [ $? -eq 0 ]; then
                                SUCCESS=$((SUCCESS+1))
                            fi
                            TOTAL=$((TOTAL+1))
                            echo "--- Progress -- : $SUCCESS / $TOTAL"
                        done
                    done
                done
            done
        done
    done
done