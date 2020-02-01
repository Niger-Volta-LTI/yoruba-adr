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
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_22500_keeper.pt

# (Nov 6th, 2019) Good model what we originally intended 22500 -- NO IROYIN
# CORPUS: {LagosNWUspeech_corpus, TheYorubaBlog_corpus, BibeliYoruba_corpus, Toluwase/Word-Level-Language-Identification-for-Resource-Scarce-, Kọ́lá Túbọ̀sún interiews}
# MODEL:  yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_22500_keeper.pt (eval'd to give the best results)
# GITHUB: https://github.com/Niger-Volta-LTI/yoruba-adr/blob/3fd97403aeb9da70e5c218feafc7d1c4edd81fed/runs/onmt/run_results_with_source_corpora.txt
# NFC OK!! because Kola's second interview isn't in here yet

# ========================================================================================
# Accuracy (%): 72.43340918217106
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_100000.pt
# Accuracy (%): 72.97166216309844
#FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_79500.pt
# Accuracy (%): 73.04896447675216
#FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_80500.pt
# Accuracy (%): 72.37981881901739
#FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_95000.pt
#Accuracy (%): 72.48129342597541
#FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_92500.pt
# Accuracy (%): 72.51117431938236
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_91500.pt
# Accuracy (%): 72.69489247311827
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_91000.pt

#Accuracy (%): 72.30301427815971
#FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_85000.pt
# Accuracy (%): 72.47736954206603
#FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_87500.pt

#Accuracy (%): 72.134166392634
#FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_88500.pt

# Accuracy (%): 72.87628375161532
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_89500.pt

# TOPBOY
# Accuracy (%): 73.34699453551913
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_90000.pt

# Accuracy (%): 73.0543470754782
# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_90500.pt

############
# Accuracy (%): 62.20488747756454
FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan31_no_JW300_EC2_step_30000.pt


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