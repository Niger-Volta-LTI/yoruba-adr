#!/usr/bin/env bash

echo "[INFO] aggregate sources from yoruba-text, split & strip to make parallel text"
./scripts/aggregate_corpora_make_parallel_text.sh


echo "[INFO] remove old tensorboard runs, and preprocessed files"
rm data/*.pt
rm -rf runs/*

echo "[INFO] preprocess training data"
python3.6 ./src/preprocess.py -train_src ./data/train/sources.txt \
                        -train_tgt ./data/train/targets.txt \
                        -valid_src ./data/dev/sources.txt \
                        -valid_tgt ./data/dev/targets.txt \
                        -save_data ./data/demo

echo "[INFO] running Bahdanau seq2seq training, for GPU training add: -gpuid 0 "
# python3.6 ./src/train.py -gpuid 0 \
python3.6 ./src/train.py \
    -data data/demo \
    -save_model models/yo_adr_bahdanau_lstm_256_1_1 \
    -tensorboard  \
    -enc_layers 1 \
    -dec_layers 1 \
    -rnn_size  128 \
    -rnn_type LSTM \
    -global_attention dot \
    -optim adam \
    -learning_rate 0.001 \
    -learning_rate_decay 0.7
