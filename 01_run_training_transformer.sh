#!/usr/bin/env bash
#
#echo "[INFO] aggregate sources from yoruba-text, split & strip to make parallel text"
#./scripts/aggregate_corpora_make_parallel_text.sh
#
#
#echo "[INFO] remove old tensorboard runs, and preprocessed files"
#rm data/*.pt
#rm -rf runs/*
#
#echo "[INFO] preprocess training data"
#python3 ./src/preprocess.py -train_src ./data/train/sources.txt \
#                        -train_tgt ./data/train/targets.txt \
#                        -valid_src ./data/dev/sources.txt \
#                        -valid_tgt ./data/dev/targets.txt \
#                        -save_data ./data/demo

echo "[INFO] running Transformer (self-attention) training, for GPU training add: -gpuid 0 "
# python3 ./src/train.py -gpuid 0 \
python3 ./src/train.py \
    -data data/demo \
    -save_model models/yo_adr_transformer_sans_yoglobalvoices_all_in \
    -save_checkpoint_steps 500 \
    -tensorboard  \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 100000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 10000

