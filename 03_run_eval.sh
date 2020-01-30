#!/bin/bash
# FILES=/Users/rebeiro/github/yoruba-adr/models/*.pt

# ========================================================================================
# Accuracy (%): 40.30612244897959
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_256_1_1_step_100000.pt

# (September 2018/March 2019) original 3 corpus Biblica, Blog, LagosNWU
# CORPUS: {LagosNWU, Bíbélì & Yorùbá blog}
# MODEL:  yo_adr_bahdanau_lstm_256_1_1_step_100000.pt
# GITHUB:
# NFC OK!!
# ========================================================================================
# Accuracy (%): 57.29068673565381
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_21000_no_kola_mistake.pt

# (Oct 16, 2019) No Kola mistake model  -- NO IROYIN
# CORPUS:  {LagosNWUspeech_corpus, TheYorubaBlog_corpus, BibeliYoruba_corpus, Toluwase/Word-Level-Language-Identification-for-Resource-Scarce-, [MISSING] Kọ́lá Túbọ̀sún interiews}
# MODEL: yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_21000_no_kola_mistake.pt
# GITHUB: https://github.com/Niger-Volta-LTI/yoruba-adr/blob/19cab1e45683e74cfe67e58b118d8d8e72368f6e/runs/onmt/run_results_with_source_corpora.txt
# NFC OK!!

# ========================================================================================
# Accuracy (%): 58.203696316979126
FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_22500_keeper.pt

# (Nov 6th, 2019) Good model what we originally intended 22500 -- NO IROYIN
# CORPUS: {LagosNWUspeech_corpus, TheYorubaBlog_corpus, BibeliYoruba_corpus, Toluwase/Word-Level-Language-Identification-for-Resource-Scarce-, Kọ́lá Túbọ̀sún interiews}
# MODEL:  yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_22500_keeper.pt (eval'd to give the best results)
# GITHUB: https://github.com/Niger-Volta-LTI/yoruba-adr/blob/3fd97403aeb9da70e5c218feafc7d1c4edd81fed/runs/onmt/run_results_with_source_corpora.txt
# NFC OK!! because Kola's second interview isn't in here yet

# ========================================================================================


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