#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Combine various files in yoruba-text, text-reserve into {train, validation, text} source & target pairs
"""
import os.path as path
import iranlowo


def main():
    """
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
    base_dir_path = path.dirname(path.dirname(path.realpath(__file__)))                         # yoruba-adr
    print("[INFO] base_dir_path: {}".format(base_dir_path))

    yoruba_text_path = path.join(path.dirname(base_dir_path), "yoruba-text")                    # yoruba-text
    print("[INFO] yoruba_text_path: {}".format(yoruba_text_path))

    yoruba_text_reserve_path = path.join(path.dirname(base_dir_path), "yoruba-text-reserve")    # yoruba-text-reserve
    print("[INFO] yoruba_text_reserve_path: {}".format(yoruba_text_reserve_path))

    # aggregated & split text paths
    aggregated_train_text_path = base_dir_path + "/data/train/combined_train.txt"
    aggregated_dev_text_path   = base_dir_path + "/data/dev/combined_dev.txt"
    aggregated_test_text_path  = base_dir_path + "/data/test/combined_test.txt"

    # Each list item has {path to file, training data offset from 0, dev offset, test offset == EOF}
    yoruba_text_corpora = [
        {"path": "LagosNWU/all_transcripts.txt",                             "train": 3883,  "dev": 432},
        {"path": "TheYorubaBlog/theyorubablog_dot_com.txt",                  "train": 3721,  "dev": 414},
        {"path": "Asubiaro_LangID/langid_combined_training_test_corpus.txt", "train": 4791,  "dev": 533},
        {"path": "Iroyin/yoglobalvoices.txt",                                "train": 557,  "dev": 61}

        # {"path": "Bibeli_Mimo/biblica.txt",                 "train": 36570, "dev": 4570, "test": 4570},
        # {"path": "Bibeli_Mimo/bsn.txt",                     "train": 3308,  "dev": 413,  "test": 414}
    ]

    yoruba_reserve_text_corpora = [
        {"path": "Kola_Tubosun_Interviews/kola_corpus.txt",  "train": 3600,  "dev": 400}
    ]

    # training, validation & test texts
    training_text = []
    dev_text = []
    test_text = []

    # Assemble public yoruba-text
    for item in yoruba_text_corpora:
        item_full_path = yoruba_text_path + "/" + item['path']
        with open(item_full_path, 'r') as f:
            x = f.read().splitlines()

        assert item['train'] + item['dev'] == len(x) - 1  # because len() is not zero based indexing

        # copy texts
        training_text += x[:item['train']]
        dev_text += x[item['train']:item['train'] + item['dev']]

    # Assemble private yoruba-text-reserve (used with permission, but not public domain, or open-source)
    for item in yoruba_reserve_text_corpora:
        item_full_path = yoruba_text_reserve_path + "/" + item['path']
        with open(item_full_path, 'r') as f:
            x = f.read().splitlines()

        assert item['train'] + item['dev'] == len(x) - 1  # because len() is not zero based indexing

        training_text += x[:item['train']]
        dev_text += x[item['train']:item['train'] + item['dev']]

    # Write files to disk
    with open(aggregated_train_text_path, 'w') as file_handler:
        for training_line in training_text:
            file_handler.write("{}\n".format(training_line))

    with open(aggregated_dev_text_path, 'w') as file_handler:
        for dev_line in dev_text:
            file_handler.write("{}\n".format(dev_line))

    print("[INFO] make parallel text dataset for yoruba diacritics restoration")

    # Write train, dev and test data
    # make_parallel_text.main(['a', 'b', 'c'])
    # --source_file ${SOURCE_FILE_TRAIN} --max_len 40 --output_dir ${OUTPUT_DIR_TRAIN}


    # ${BASE_DIR}/src/make_parallel_text.py --source_file ${SOURCE_FILE_DEV} --max_len 40 --output_dir ${OUTPUT_DIR_DEV}

    # ${BASE_DIR}/src/make_parallel_text.py --source_file ${SOURCE_FILE_TEST} --max_len 40 --output_dir ${OUTPUT_DIR_TEST}



    # # clean up intermediates, to leave only final parallel text {sources.txt, targets.txt}
    # rm ${SOURCE_FILE_TRAIN} ${SOURCE_FILE_DEV} ${SOURCE_FILE_TEST}
    #
    #


    # # Generate dataset
    # examples = list(make_data(ARGS.source_file, ARGS.min_len, ARGS.max_len))
    # try:
    #     os.makedirs(ARGS.output_dir)
    # except OSError:
    #     if not os.path.isdir(ARGS.output_dir):
    #         raise
    #
    # # Write train data
    # train_sources, train_targets = zip(*examples)
    # write_parallel_text(train_sources, train_targets, ARGS.output_dir)

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


if __name__ == "__main__":
    main()