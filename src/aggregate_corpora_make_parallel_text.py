#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Combine various files in yoruba-text, text-reserve into {train, validation, text} source & target pairs

    We expect that yoruba-text, yoruba-adr & yoruba-text-reserve (for those who have access) are SIBLING
    directories, within a parent directory. For example: /home/iroro/github

    ${HOME}                         /home/iroro/
    ├── Desktop                     /home/iroro/Desktop/
    ├── Downloads                   /home/iroro/Downloads/
    ├── github                      /home/iroro/github/                 <--- where all code lives
    │   ├── yoruba-adr              /home/iroro/github/yoruba-adr/
    │   │   ├── ...
    │   ├── yoruba-text             /home/iroro/github/yoruba-text/
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
    {"path": "LagosNWU/all_transcripts.txt",            "train": 3452,  "dev": 431,  "test": 432},
    {"path": "TheYorubaBlog/theyorubablog_dot_com.txt", "train": 3308,  "dev": 413,  "test": 414},
    # {"path": "Bibeli_Mimo/biblica.txt",                 "train": 36570, "dev": 4570, "test": 4570},
    # {"path": "Bibeli_Mimo/bsn.txt",                     "train": 3308,  "dev": 413,  "test": 414}
    # {"path": "Iroyin/news_sites.txt",                   "train": 1371,  "dev": 171,  "test": 171}
]

# training, validation & test texts
training_text = []
dev_text = []
test_text = []

for item in yoruba_text_paths:
    item_full_path = yoruba_text_path + "/" + item['path']

    # read in item
    with open(item_full_path, 'r') as f:
        x = f.read().splitlines()

    # assert item['train'] + item['dev'] + item['test'] == len(x)
    print(item['train'] + item['dev'] + item['test'])
    print(len(x))

    # copy item['train'] items to train_text
    # copy item['dev'] items to dev_text
    # copy item['test' items to test_text



# for path in yoruba_text_reserve_paths:



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