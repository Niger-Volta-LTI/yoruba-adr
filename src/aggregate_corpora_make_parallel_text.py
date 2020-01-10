#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Combine various files in yoruba-text, text-reserve into {train, validation, text} source & target pairs
"""

import codecs
import io
import iranlowo.adr as ránlọ
import os.path as path
import os
import re

from nltk.tokenize import word_tokenize


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
    test_text_path             = base_dir_path + "/data/test/news_sites.targets.txt"

    try:

        os.makedirs(path.split(path.abspath(aggregated_train_text_path))[0])
        os.makedirs(path.split(path.abspath(aggregated_dev_text_path))[0])
        os.makedirs(path.split(path.abspath(test_text_path))[0])
    except OSError:
        print("[ERROR] could not make paths for combined text")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Each list item has {path to file, training data offset from 0, dev offset ==> EOF}
    yoruba_text_corpora = [
        {"path": "LagosNWU/all_transcripts.txt",                             "train": 3883,  "dev": 432},
        {"path": "TheYorubaBlog/theyorubablog_dot_com.txt",                  "train": 3721,  "dev": 414},
        {"path": "Asubiaro_LangID/langid_combined_training_test_corpus.txt", "train": 4791,  "dev": 533},
        {"path": "Bibeli_Mimo/biblica.txt",                                  "train": 27738, "dev": 3083},
        {"path": "Bibeli_Mimo/bsn.txt",                                      "train": 29033, "dev": 3226},
        {"path": "Owe/owe.txt",                                              "train": 2429,  "dev": 271},
        {"path": "JW300/jw300.yo.txt",                                       "train": 470237, "dev": 4749},
        {"path": "Universal_Declaration_Human_Rights/unhr.yo.txt",           "train": 134,    "dev": 15},
        {"path": "Alabi_YorubaTwi_Embedding/alakowe.txt",                    "train": 648, "dev": 72},
        {"path": "Alabi_YorubaTwi_Embedding/edeyorubarewa.txt",              "train": 114, "dev": 13},
        {"path": "Alabi_YorubaTwi_Embedding/igbo_olodumare.txt",             "train": 533, "dev": 60},
        {"path": "Alabi_YorubaTwi_Embedding/irinkerindo_ninu_igbo.txt",      "train": 621, "dev": 70},
        {"path": "Alabi_YorubaTwi_Embedding/oroyoruba.txt",                  "train": 1434, "dev": 160}
    ]

    yoruba_reserve_text_corpora = [
        {"path": "Kola_Tubosun/Interviews/kola_corpus.txt",    "train": 3479,  "dev": 387},
        {"path": "Kola_Tubosun/201906_corpus/Kola_201906.txt", "train": 629,   "dev": 70},
        {"path": "Timi_Wuraola/HÁÀ_ÈNÌYÀN_edited_wuraola.txt", "train": 1499,  "dev": 167}
    ]

    # Lexicon and texts derived from computing most common trigrams, bigrams, etc
    yoruba_lesika = [
        {"path": "Lesika/yoruba_words_sorted.txt",                           "train": 41584}
    ]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Iroyin yoglobalvoices test/evaluation dataset
    yoruba_evaluation_dataset = [
        # {"path": "Iroyin/news_sites.txt",                                "test": 1738}
        {"path": "Iroyin/yoglobalvoices.txt",                              "test": 618}
    ]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # training, validation & test texts
    training_text = []
    dev_text = []
    test_text = []

    # Assemble public yoruba-text
    assemble_text_from(yoruba_text_corpora, yoruba_text_path, training_text, dev_text)

    # Assemble private yoruba-text-reserve
    assemble_text_from(yoruba_reserve_text_corpora, yoruba_text_reserve_path, training_text, dev_text)

    # training_text += yoruba_lesika
    item_full_path = yoruba_text_path + "/" + yoruba_lesika[0]['path']
    with open(item_full_path, 'r') as f:
        x = f.read().splitlines()
        training_text += x

    # read in text_text (so we can grab unigrams from it)
    for item in yoruba_evaluation_dataset:
        item_full_path = yoruba_text_path + "/" + item['path']
        with open(item_full_path, 'r') as f:
            x = f.read().splitlines()
        test_text += x

    # check all text
    all_text = training_text + dev_text + test_text
    print("training text len: " + str(len(training_text)))
    print("dev text len: " + str(len(dev_text)))
    print("test text len: " + str(len(test_text)))
    print("all_text len: " + str(len(all_text)))

    # make all_text to derive ngrams from
    counts = {}
    i = 0
    for line in all_text:
        if line.strip() == '':  # empty line case
            continue

        # clean up
        line = sanitize_line(line)

        word_array = line.split()
        if len(word_array) > 1:
            for word in word_array:
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1
        else:
            word = line         # single word in a line case
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1
        i+=1

    print("Word count == " + str(len(counts)))
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    # for word, frequency in sorted_counts:
    #     print("%s %d" % (word, frequency))

    # Add all_text-derived Lesika to training text
    training_text += list(counts.keys())


    ######################################################################

    # Write files to disk
    with open(aggregated_train_text_path, 'w') as file_handler:
        for training_line in training_text:
            file_handler.write("{}\n".format(training_line))

    with open(aggregated_dev_text_path, 'w') as file_handler:
        for dev_line in dev_text:
            file_handler.write("{}\n".format(dev_line))

    print("[INFO] make parallel text dataset for yoruba diacritics restoration")
    make_parallel_text(aggregated_train_text_path, base_dir_path + "/data/train/", 5, 100)
    make_parallel_text(aggregated_dev_text_path, base_dir_path + "/data/dev/", 5, 100)
    make_parallel_text(test_text_path, base_dir_path + "/data/test/", 5, 100)


def assemble_text_from(yoruba_text_corpora, yoruba_text_path, training_text, dev_text):
    for item in yoruba_text_corpora:
        item_full_path = yoruba_text_path + "/" + item['path']
        with open(item_full_path, 'r') as f:
            x = f.read().splitlines()

        print(item['path'], str(len(x) - 1))
        assert item['train'] + item['dev'] == len(x) - 1  # because len() is not zero based indexing

        # copy texts
        training_text += x[:item['train']]
        dev_text += x[item['train']:item['train'] + item['dev']]


forbidden_symbols = re.compile(r"[\[\]\(\)\/\\\>\<\=\+\_\*\”\“\‘]")
numbers = re.compile(r"\d")
multiple_punct = re.compile(r"([\.\?\!\,\:\;\-])(?:[\.\?\!\,\:\;\-]){1,}")
is_number = lambda x: len(numbers.sub("", x)) / len(x) < 0.6
NUM = "<NUM>"


def sanitize_line(line):
    lower_line = line.lower()
    # return re.sub(r'[^a-zA-Z_\s]+', '', lower_line)
    return re.sub(r'\W+', ' ', lower_line)

def skip(line):
    # collapse empty lines
    if line.strip() == "":
        return True

    # skip forbidden symbols
    if forbidden_symbols.search(line) is not None:
        return True

    return False


def truncate_string(line, number_of_words):
    return " ".join(line.split()[:number_of_words])


def make_parallel_text(source_file, output_dir, min_len, max_len):
    # Generate dataset
    examples = list(make_data(source_file, min_len, max_len))
    try:
        os.makedirs(output_dir)
    except OSError:
        if not path.isdir(output_dir):
            raise

    # Write train data
    train_sources, train_targets = zip(*examples)
    write_parallel_text(train_sources, train_targets, output_dir)


def make_data(source_file, min_len, max_len):
    """
    Generates a dataset for yoruba diacritics restoration
    Sequence lengths are chosen to be on in [min_len, max_len]

    Args:
      source_file: diacritized source file ==> source & target tokenized files
      num_examples: Number of examples to generate
      min_len: Minimum sequence length
      max_len: Maximum sequence length

    Returns:
      An iterator of (source, target) string tuples.
    """
    # since we don't have a lot of data, we should take all the data in the file
    # which means pre-splitting before sending into make data

    skipped = 0
    with codecs.open(source_file, "r", "utf-8") as text:
        for line in text:
            line1 = line.replace('"', "").replace(",", "").strip()

            # if line1 != line:
            #     print("line1 != line")

            # flatten multiple punctuations (??) into a single token
            line2 = multiple_punct.sub(r"\g<1>", line1)
            if line2 != line1:
                print("line2 != line1")

            if skip(line2):
                skipped += 1
                print("Skipping: " + line2)
                continue

            if len(line2.split()) < min_len:
                continue

            line3 = truncate_string(line2, max_len)  # truncate to max_len
            target_tokens = word_tokenize(line3.lower())

            output_tokens = []

            for token in target_tokens:
                if is_number(token):
                    output_tokens.append(NUM)
                else:
                    output_tokens.append(token.lower())

            # yield source_tokens = strip_accents(target_tokens)
            yield " ".join([ránlọ.strip_accents_text(token) for token in target_tokens]), " ".join(
                target_tokens
            )


def write_parallel_text(sources, targets, output_prefix):
    """
    Writes two files where each line corresponds to one example
      - [output_prefix].sources.txt
      - [output_prefix].targets.txt

    Args:
      sources: Iterator of source strings
      targets: Iterator of target strings
      output_prefix: Prefix for the output file
    """
    source_filename = path.abspath(path.join(output_prefix, "sources.txt"))
    target_filename = path.abspath(path.join(output_prefix, "targets.txt"))

    with io.open(source_filename, "w", encoding="utf8") as source_file:
        for record in sources:
            source_file.write(record + "\n")
    print("Wrote {}".format(source_filename))

    with io.open(target_filename, "w", encoding="utf8") as target_file:
        for record in targets:
            target_file.write(record + "\n")
    print("Wrote {}".format(target_filename))


if __name__ == "__main__":
    main()
