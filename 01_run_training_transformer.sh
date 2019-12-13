#!/usr/bin/env bash

echo "[INFO] aggregate sources from yoruba-text, split & strip to make parallel text"
./scripts/aggregate_corpora_make_parallel_text.sh


echo "[INFO] remove old tensorboard runs, and preprocessed files"
rm data/*.pt
rm -rf runs/*

echo "[INFO] preprocess training data"
python3 ./src/preprocess.py -train_src ./data/train/sources.txt \
                        -train_tgt ./data/train/targets.txt \
                        -valid_src ./data/dev/sources.txt \
                        -valid_tgt ./data/dev/targets.txt \
                        -save_data ./data/demo

echo "[INFO] running Transformer (self-attention) training, for GPU training add: -gpuid 0 "
# python3 ./src/train.py -gpuid 0 \
python3 ./src/train.py \
    -data data/demo \
    -save_model models/yo_adr_transformer \
    -save_checkpoint_steps 500 \
    -tensorboard  \
    -encoder_type transformer \
    -decoder_type transformer \
    -enc_layers 2 \
    -dec_layers 2 \
    -rnn_size 512 \
    -word_vec_size 512 \
    -global_attention dot \
    -optim adam \
    -learning_rate 0.001 \
    -learning_rate_decay 0.7