#!/bin/bash
FILES=/Users/iroro/github/yoruba-adr/models/*.pt

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