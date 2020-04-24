
TEST SET: Global Voices from David

# ========================================================================================
# (September 2018/March 2019) original ADR Bahdanau soft-attention model
# PRED PPL: 1.3390
# >> BLEU = 26.53, 55.2/35.1/23.7/16.3 (BP=0.901, ratio=0.906, hyp_len=14846, ref_len=16391
# WER: 0.5816519353855532

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_256_1_1_step_100000.pt
# CORPUS: {LagosNWU, Bíbélì & Yorùbá blog}

# ========================================================================================
# (Oct 16, 2019) Bahdanau soft-attention -> Enhanced model from more data - Forgot Kola data - mistake model -- NO IROYIN
#  PRED PPL: 1.6884
# >> BLEU = 42.52, 68.1/49.0/36.3/27.3 (BP=0.997, ratio=0.997, hyp_len=16344, ref_len=16391)
# WER: 0.33026516306004267

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_21000_no_kola_mistake.pt
# CORPUS:  {LagosNWUspeech_corpus, TheYorubaBlog_corpus, BibeliYoruba_corpus, Toluwase/Word-Level-Language-Identification-for-Resource-Scarce-, [MISSING] Kọ́lá Túbọ̀sún interiews}

# ========================================================================================
# (Nov 6th, 2019) Bahdanau soft-attention -> Enhanced model from more data - NO IROYIN
# PRED PPL: 1.5862
# >> BLEU = 42.23, 68.5/48.6/35.9/26.6 (BP=1.000, ratio=1.000, hyp_len=16391, ref_len=16391)
# WER: 0.3257543431880524

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_iroyin_step_22500_keeper.pt
# CORPUS: {LagosNWUspeech_corpus, TheYorubaBlog_corpus, BibeliYoruba_corpus, Toluwase/Word-Level-Language-Identification-for-Resource-Scarce-, Kọ́lá Túbọ̀sún interiews}

# ========================================================================================
# SOFT ATTENTION - ALL NEW DATA
# ========================================================================================
(Jan 30, 2020) Bahdanau soft-attention model
# PRED PPL: 1.4415
# >> BLEU = 59.55, 80.2/65.4/53.7/44.7 (BP=1.000, ratio=1.000, hyp_len=16390, ref_len=16391)
# WER: 0.20396220664431575

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan30_EC2_step_90000.pt
# CORPUS: all-new-data, no Global Voices

# ========================================================================================
# (Jan 31, 2020) Bahdanau soft-attention model without JW300
# PRED PPL: 1.6007
# >> BLEU = 43.39, 68.9/49.7/37.1/28.0 (BP=1.000, ratio=1.000, hyp_len=16388, ref_len=16391)
# WER: 0.3186833282535812

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_jan31_no_JW300_EC2_step_23500.pt
# CORPUS: all-new-data, no Global Voices, no JW300

# ========================================================================================
# (Feb9, 2020) Bahdanau soft-attention model + FastText Embedding
#  PRED PPL: 1.3923
# >> BLEU = 58.87, 79.8/64.7/53.1/44.1 (BP=0.998, ratio=0.998, hyp_len=16362, ref_len=16391)
# WER: 0.21334958854007924

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_bahdanau_lstm_128_2_2_sans_yoglobalvoices_all_in_take3_feb9_fasttext_EC2_step_91500.pt
# CORPUS: all-new-data, no Global Voices

# ========================================================================================
# TRANSFORMER - ALL NEW DATA
# ========================================================================================
# (Feb2, 2020) TRANSFORMER
# PRED PPL: 1.4002
# >> BLEU = 59.05, 80.6/66.6/55.9/47.3 (BP=0.962, ratio=0.963, hyp_len=15779, ref_len=16391)
# WER: 0.23097080870254127

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb2_EC2_step_21500.pt
# CORPUS: all-new-data, no Global Voices

# ========================================================================================
# (Feb1, 2020) TRANSFORMER without JW300
# PRED PPL: 1.9456
# >> BLEU = 45.68, 74.6/57.5/45.1/36.1 (BP=0.889, ratio=0.894, hyp_len=14659, ref_len=16391)
# WER: 0.3439785470502194

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb1_no_JW300_EC2_step_15500.pt
# CORPUS: all-new-data, no Global Voices, no JW300

# ========================================================================================
# (Feb9, 2020) TRANSFORMER+ FastText Embedding
# PRED PPL: 1.4250
# >> BLEU = 59.80, 81.8/67.0/56.5/47.9 (BP=0.964, ratio=0.965, hyp_len=15815, ref_len=16391)
# WER: 0.22426604945791204

# FILES=/Users/iroro/github/yoruba-adr/models/yo_adr_transformer_sans_yoglobalvoices_all_in_take3_feb8_fasttext_EC2_step_18500.pt
# CORPUS: all-new-data, no Global Voices