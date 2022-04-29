#!/usr/bin/env bash
source ../MML/virtual/bin/activate
rm tfidf_baselines
touch tfidf_baselines


# AMBIGUITY CASES
GRAM_LOC=../Representation_Probing/Representations/Ambiguity
./src/classification_baseline.py -d $GRAM_LOC/BERT_COCO/sentence_representations.pkl --target amb
./src/classification_baseline.py -d $GRAM_LOC/BERT_EMMT/sentence_representations.pkl --target amb

# GRAMATICALITY CASES
GRAM_LOC=../Representation_Probing/Representations/Grammaticality
for d in $GRAM_LOC/*; do
    n_loc=$d
    bert_repr=$n_loc/bert_sentence_representations.pkl
    
    if [ -f "$bert_repr" ]; then
        echo $n_loc
        ./src/classification_baseline.py -d $bert_repr --target class 
    
    else
        for case in $n_loc/*/; do
            echo $case
            bert_repr=$case"bert_sentence_representations.pkl"
            ./src/classification_baseline.py -d $bert_repr --target class
        done

    fi
done

