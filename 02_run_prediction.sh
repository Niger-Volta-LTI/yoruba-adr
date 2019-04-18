#!/usr/bin/env bash

echo "[INFO] running inference on test sources"

# Note that in lieu of: -model models/yo_adr_bahdanau_lstm_256_1_1_step_1000.pt
# put the path to your model checkpoint or final model
python3.6 ./src/translate.py \
	-model models/yo_adr_bahdanau_lstm_256_1_1_step_100000.pt \
	-src data/test/one_phrase.txt \
	-output data/test/pred.txt \
	-replace_unk \
	-verbose