#!/bin/bash
# FILES=/Users/rebeiro/github/yoruba-adr/models/*.pt

### OLDER
# Accuracy (%): 18.318043811824623
# FILES=/Users/rebeiro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_256_1_1_step_100000.pt

# Accuracy (%): 48.2125950834122 (does this have yoglobalvoices in it?)
# FILES=/Users/rebeiro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_21000_no_kola_mistake.pt

# Accuracy (%): 46.10362051516438 includes yoglobalvoices
# FILES=/Users/rebeiro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_all_in_take1_step_100000.pt

# Accuracy (%): 88.39084587509386 includes yoglobalvoices
# FILES=/Users/rebeiro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_step_90000_release.pt

## ----
# Accuracy (%): 41.26538474094871 -- Transformer!!
# FILES=/Users/rebeiro/github/yoruba-adr/models/yo_adr_transformer_sans_yoglobalvoices_all_in_step_6000.pt

# Accuracy (%): 45.643302536645876  Seq2Seq
FILES=/Users/rebeiro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take2_step_100000.pt


for f in $FILES
do
  echo "Processing $f file..."

  # take action on each file. $f store current file name
  python3 ./src/translate.py \
	-model $f \
	-src data/test/sources.txt \
	-tgt data/test/targets.txt \
	-output data/test/pred.txt \
	-replace_unk \
	-verbose
done