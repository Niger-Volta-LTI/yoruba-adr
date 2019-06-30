#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Combine various files in yoruba-text, text-reserve into {train, validation, text} source & target pairs

    We expect that yoruba-text, yoruba-adr & yoruba-text-reserve (for those who have access) are SIBLING
    directories, within a parent directory like github, for example: /home/iroro/github

    ${HOME}                         /home/iroro/
    ├── Desktop                     /home/iroro/Desktop/
    ├── Downloads                   /home/iroro/Downloads/
    ├── github                      /home/iroro/github/                 <--- where all code lives
    │   ├── yoruba-text             /home/iroro/github/yoruba-text/
    │   │   ├── ...
    │   ├── yoruba-adr              /home/iroro/github/yoruba-adr/
    │   │   ├── ...
    │   ├── yoruba-text-reserve     /home/iroro/github/yoruba-text-reserve
    │   │   └── ...

"""
import os.path as path


base_dir_path = path.dirname(path.dirname(path.realpath(__file__)))                         # yoruba-adr
print(base_dir_path)

yoruba_text_path = path.join(path.dirname(base_dir_path), "yoruba-text")                    # yoruba-text
print(yoruba_text_path)

yoruba_text_reserve_path = path.join(path.dirname(base_dir_path), "yoruba-text-reserve")    # yoruba-text-reserve
print(yoruba_text_reserve_path)


# Each list item has {path to file, training data offset from 0, dev offset, test offset == EOF}
yoruba_text_paths = [
    {"path": "LagosNWU/all_transcripts.txt",            "train": 3452, "dev": 431, "test": 432},
    {"path": "TheYorubaBlog/theyorubablog_dot_com.txt", "train": 3308, "dev": 413, "test": 414}
]


#
# # setup output dirs with train/dev/test splits
# OUTPUT_DIR_TRAIN="${OUTPUT_DIR}/train"
# OUTPUT_DIR_DEV="${OUTPUT_DIR}/dev"
# OUTPUT_DIR_TEST="${OUTPUT_DIR}/test"
#
# # start afresh each time
#
#
# SOURCE_FILE_TRAIN="${OUTPUT_DIR_TRAIN}/train.txt"
# SOURCE_FILE_DEV="${OUTPUT_DIR_DEV}/dev.txt"
# SOURCE_FILE_TEST="${OUTPUT_DIR_TEST}/test.txt"
#
# ###############################################################################################################
# ### FOR LagosNWUspeech_corpus: 4315 lines => 80/10/10 split => train/dev/test => 3452/431/432
# echo ""
# echo "Using [LagosNWUspeech] SOURCE FILE TRAIN=${SOURCE_FILE_TRAIN}"
# head -n 3452 "${SOURCE_BASE_DIR}/LagosNWU/all_transcripts.txt" >>  ${SOURCE_FILE_TRAIN}
#
# echo "Using [LagosNWUspeech] SOURCE FILE TRAIN=${SOURCE_FILE_DEV}"
# tail -n 863 "${SOURCE_BASE_DIR}/LagosNWU/all_transcripts.txt" | head -n 431  >> ${SOURCE_FILE_DEV}
#
# echo "Using [LagosNWUspeech] SOURCE FILE TRAIN=${SOURCE_FILE_TEST}"
# tail -n 863 "${SOURCE_BASE_DIR}/LagosNWU/all_transcripts.txt" | tail -n 432  >> ${SOURCE_FILE_TEST}
# echo "" >> ${SOURCE_FILE_TEST}
#
#
# ###############################################################################################################
# ### FOR TheYorubaBlog_corpus: 4135 lines => 80/10/10 split => train/dev/test => 3308/413/414
# echo ""
# echo "Using [TheYorubaBlog] SOURCE FILE TRAIN=${SOURCE_FILE_TRAIN}"
# head -n 3308 "${SOURCE_BASE_DIR}/TheYorubaBlog/theyorubablog_dot_com.txt" >>  ${SOURCE_FILE_TRAIN}
#
# echo "Using [TheYorubaBlog] SOURCE FILE TRAIN=${SOURCE_FILE_DEV}"
# tail -n 827 "${SOURCE_BASE_DIR}/TheYorubaBlog/theyorubablog_dot_com.txt" | head -n 413  >> ${SOURCE_FILE_DEV}
#
# echo "Using [TheYorubaBlog] SOURCE FILE TRAIN=${SOURCE_FILE_TEST}"
# tail -n 827 "${SOURCE_BASE_DIR}/TheYorubaBlog/theyorubablog_dot_com.txt" | tail -n 414  >> ${SOURCE_FILE_TEST}
# echo "" >> ${SOURCE_FILE_TEST}
#
#
# ###############################################################################################################
# ### FOR BibeliYoruba_corpus: 45713 lines => 80/10/10 split => train/dev/test => 36570/4570/4570
# echo ""
# echo "Using [BibeliYoruba] SOURCE FILE TRAIN=${SOURCE_FILE_TRAIN}"
# head -n 36570 "${SOURCE_BASE_DIR}/Bibeli_Mimo/bibeli_ede_yoruba.txt" >>  ${SOURCE_FILE_TRAIN}
#
# echo "Using [BibeliYoruba] SOURCE FILE TRAIN=${SOURCE_FILE_DEV}"
# tail -n 9143 "${SOURCE_BASE_DIR}/Bibeli_Mimo/bibeli_ede_yoruba.txt" | head -n 4571  >> ${SOURCE_FILE_DEV}
#
# echo "Using [BibeliYoruba] SOURCE FILE TRAIN=${SOURCE_FILE_TEST}"
# tail -n 9143 "${SOURCE_BASE_DIR}/Bibeli_Mimo/bibeli_ede_yoruba.txt" | tail -n 4571  >> ${SOURCE_FILE_TEST}
# echo "" >> ${SOURCE_FILE_TEST}
#
#
# ###############################################################################################################
# ### FOR Iroyin (news): 1738 lines => 80/10/10 split => train/dev/test => 1371/171/171
# echo ""
# echo "Using [Iroyin] SOURCE FILE TRAIN=${SOURCE_FILE_TRAIN}"
# head -n 1371 "${SOURCE_BASE_DIR}/Iroyin/news_sites.txt" >>  ${SOURCE_FILE_TRAIN}
#
# echo "Using [Iroyin] SOURCE FILE TRAIN=${SOURCE_FILE_DEV}"
# tail -n 342 "${SOURCE_BASE_DIR}/Iroyin/news_sites.txt" | head -n 171  >> ${SOURCE_FILE_DEV}
#
# echo "Using [Iroyin] SOURCE FILE TRAIN=${SOURCE_FILE_TEST}"
# tail -n 342 "${SOURCE_BASE_DIR}/Iroyin/news_sites.txt" | tail -n 171  >> ${SOURCE_FILE_TEST}
# echo "" >> ${SOURCE_FILE_TEST}
#
#
# ############################################################################################################
# ### FOR https://github.com/Toluwase/Word-Level-Language-Identification-for-Resource-Scarce-
# ### A corpus for word-level language id research:  5324 lines => 80/10/10 => train/dev/test => 4258/533/533
#
# SOURCE_BASE_DIR="${BASE_DIR}/../Word-Level-Language-Identification-for-Resource-Scarce-"
# echo ""
# echo "Changing to use SOURCE_TEXT_BASE_DIR=${SOURCE_BASE_DIR}"
#
# ### Check if this repo exists, git clone it if it doesn't, for now assume it does ###
# if [ -d "${SOURCE_BASE_DIR}" ]
# then
#     # cat this repo's {training, test} files together {Yoruba_training_corpus(part).txt, EngYor_test_corpus.txt}
#     cat "${SOURCE_BASE_DIR}/Yoruba_training_corpus(part).txt" "${SOURCE_BASE_DIR}/EngYor_test_corpus.txt" > "${SOURCE_BASE_DIR}/combined_corpus.txt"
#
#     echo "Using [Tolúwaṣẹ word-level langid] SOURCE FILE TRAIN=${SOURCE_FILE_TRAIN}"
#     head -n 4258 "${SOURCE_BASE_DIR}/combined_corpus.txt" >>  ${SOURCE_FILE_TRAIN}
#
#     echo "Using [Tolúwaṣẹ word-level langid] SOURCE FILE TRAIN=${SOURCE_FILE_DEV}"
#     tail -n 1066 "${SOURCE_BASE_DIR}/combined_corpus.txt" | head -n 533  >> ${SOURCE_FILE_DEV}
#
#     echo "Using [Tolúwaṣẹ word-level langid] SOURCE FILE TRAIN=${SOURCE_FILE_TEST}"
#     tail -n 1066 "${SOURCE_BASE_DIR}/combined_corpus.txt" | tail -n 533  >> ${SOURCE_FILE_TEST}
#     echo "" >> ${SOURCE_FILE_TEST}
#     echo "Removing Tempfile ${SOURCE_BASE_DIR}/Yoruba_training_corpus(part).txt"
# fi
#
#
# ############################################################################################################
# ### FOR Kọ́lá Túbọ̀sún interiews: 4001 lines => 80/10/10 split => train/dev/test => 3201/400/400
#
# SOURCE_BASE_DIR="${BASE_DIR}/../yoruba-text-reserve"
# echo ""
# echo "Changing to use SOURCE_TEXT_BASE_DIR=${SOURCE_BASE_DIR}"
#
# ### Check if text-reserve exists, which it will for users with permission ###
# if [ -d "${SOURCE_BASE_DIR}" ]
# then
#     echo "Using [Kọ́lá Túbọ̀sún interviews] SOURCE FILE TRAIN=${SOURCE_FILE_TRAIN}"
#     head -n 3201 "${SOURCE_BASE_DIR}/Kola_Tubosun_Interviews/kola_corpus.txt" >>  ${SOURCE_FILE_TRAIN}
#
#     echo "Using [Kọ́lá Túbọ̀sún interviews] SOURCE FILE TRAIN=${SOURCE_FILE_DEV}"
#     tail -n 800 "${SOURCE_BASE_DIR}/Kola_Tubosun_Interviews/kola_corpus.txt" | head -n 400  >> ${SOURCE_FILE_DEV}
#
#     echo "Using [Kọ́lá Túbọ̀sún interviews] SOURCE FILE TRAIN=${SOURCE_FILE_TEST}"
#     tail -n 800 "${SOURCE_BASE_DIR}/Kola_Tubosun_Interviews/kola_corpus.txt" | tail -n 400  >> ${SOURCE_FILE_TEST}
#     echo "" >> ${SOURCE_FILE_TEST}
# fi
#
#
# # each list item has {path to file, training data offset from 0, dev offset, test offset == EOF}
# corpora_to_process = [ {},
#
#
#
#
# ]