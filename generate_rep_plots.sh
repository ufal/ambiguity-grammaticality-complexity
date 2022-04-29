#!/usr/bin/env bash
source ../MML/virtual/bin/activate
SBERT="SBERT"
BERT="BERT"


Amb_LOC=../Representation_Probing/Representations/Ambiguity
rep_file=sentence_representations.pkl
for d in $Amb_LOC/*; do
    n_loc=$d
    FILENAME="$(basename $d)"
    arrIN=(${FILENAME//_/ })
    item=${arrIN[0]}
    repr=$n_loc/$rep_file
    ./src/classification_mlp.py --data $repr --target amb
    FILENAME="$(basename $n_loc)"
    ./src/fig_classification.py --data computed/mlp_$FILENAME.json --target amb 
done




GRAM_LOC=../Representation_Probing/Representations/Grammaticality
for d in $GRAM_LOC/*; do
    n_loc=$d
    bert_repr=$n_loc/bert_sentence_representations.pkl
    gpt_repr=$n_loc/gpt_sentence_representations.pkl
    sbert_repr=$n_loc/sbert_sentence_representations.pkl
    if [ -f "$bert_repr" ]; then
        FILENAME="$(basename $d)"
        ./src/classification_mlp.py --data $bert_repr --target class
        ./src/fig_classification.py --data computed/mlp_BERT_$FILENAME.json --target class
        ./src/classification_mlp.py --data $gpt_repr --target class
        ./src/fig_classification.py --data computed/mlp_GPT_$FILENAME.json --target class
        ./src/classification_mlp.py --data $sbert_repr --target class
        ./src/fig_classification.py --data computed/mlp_SBERT_$FILENAME.json --target class
    else
        for case in $n_loc/*/; do
            FILENAME="$(basename $case)"
            bert_repr=$case"bert_sentence_representations.pkl"
            gpt_repr=$case"bert_sentence_representations.pkl"
            sbert_repr=$case"bert_sentence_representations.pkl"
            ./src/classification_mlp.py --data $bert_repr --target class
            ./src/fig_classification.py --data computed/mlp_BERT_$FILENAME.json --target class
            ./src/classification_mlp.py --data $gpt_repr --target class
            ./src/fig_classification.py --data computed/mlp_GPT_$FILENAME.json --target class
            ./src/classification_mlp.py --data $sbert_repr --target class
            ./src/fig_classification.py --data computed/mlp_SBERT_$FILENAME.json --target class
        done
    fi
done



