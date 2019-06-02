#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 iroro orife.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functions to generate a parallel text dataset for Yorùbá diacritic restoration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import word_tokenize

import numpy as np
import unicodedata
import argparse
import codecs
import os
import io
import sys
import re

PARSER = argparse.ArgumentParser(description="Generates yoruba diacritic datasets.")
PARSER.add_argument("--source_file", type=str, help="path to source diacritics file", required=True)
PARSER.add_argument("--min_len", type=int, default=1, help="minimum sequence length")
PARSER.add_argument("--max_len", type=int, default=40, help="maximum sequence length")
PARSER.add_argument("--output_dir", type=str, help="path to the output directory", required=True)
ARGS = PARSER.parse_args()

forbidden_symbols = re.compile(r"[\[\]\(\)\/\\\>\<\=\+\_\*]")
numbers = re.compile(r"\d")
multiple_punct = re.compile(r"([\.\?\!\,\:\;\-])(?:[\.\?\!\,\:\;\-]){1,}")
is_number = lambda x: len(numbers.sub("", x)) / len(x) < 0.6
NUM = "<NUM>"


# IO HAVOC -- replace with Ìrànlọ́wọ́
def strip_accents(string):
    """
    Removes diacritics from characters, ascii-fication

    Args:
      string: diacritics string to strip

    Returns:
      ascii-fied string
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", string)
        if unicodedata.category(c) != "Mn"
    )


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
            yield " ".join([strip_accents(token) for token in target_tokens]), " ".join(
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
    source_filename = os.path.abspath(os.path.join(output_prefix, "sources.txt"))
    target_filename = os.path.abspath(os.path.join(output_prefix, "targets.txt"))

    with io.open(source_filename, "w", encoding="utf8") as source_file:
        for record in sources:
            source_file.write(record + "\n")
    print("Wrote {}".format(source_filename))

    with io.open(target_filename, "w", encoding="utf8") as target_file:
        for record in targets:
            target_file.write(record + "\n")
    print("Wrote {}".format(target_filename))


def main():
    # Generate dataset
    examples = list(make_data(ARGS.source_file, ARGS.min_len, ARGS.max_len))
    try:
        os.makedirs(ARGS.output_dir)
    except OSError:
        if not os.path.isdir(ARGS.output_dir):
            raise

    # Write train data
    train_sources, train_targets = zip(*examples)
    write_parallel_text(train_sources, train_targets, ARGS.output_dir)


if __name__ == "__main__":
    main()
