#!/usr/bin/bash

mkdir -p data_tmp

if [ $# -lt 2 ]; then
    SRC_NAME="CoLA"
    TGT_NAME="CoLA"
else
    SRC_NAME=$1
    TGT_NAME=$2
fi

echo "$SRC_NAME -> $TGT_NAME"

scp -r athena:/home/bhattacharya/personal_work_troja/Representation_Probing/Representations/Grammaticality/$SRC_NAME/* data_tmp

mv data_tmp/bert_sentence_representations.pkl data/$TGT_NAME\_BERT.pkl
mv data_tmp/sbert_sentence_representations.pkl data/$TGT_NAME\_SBERT.pkl
mv data_tmp/gpt_sentence_representations.pkl data/$TGT_NAME\_GPT.pkl

# mv data_tmp/BERT_EMMT/sentence_representations.pkl data/$TGT_NAME\_EMMT_BERT.pkl
# mv data_tmp/SBERT_EMMT/sentence_representations.pkl data/$TGT_NAME\_EMMT_SBERT.pkl
# mv data_tmp/GPT_EMMT/sentence_representations.pkl data/$TGT_NAME\_EMMT_GPT.pkl

rm -rf data_tmp