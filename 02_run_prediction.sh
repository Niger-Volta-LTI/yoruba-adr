#!/usr/bin/env bash

echo "[INFO] running inference on test sources"

# Note that in lieu of: -model models/yo_adr_bahdanau_lstm_256_1_1_step_1000.pt
# put the path to your model checkpoint or final model
python3 ./src/translate.py \
	-model models/yo_adr_bahdanau_lstm_128_2_2_step_90000_release.pt \
	-src data/test/one_phrase.txt \
	-tgt data/test/one_phrase.target.txt \
	-output data/test/pred.txt \
	-replace_unk \
	-verbose