#!/usr/bin/env bash
source ../MML/virtual/bin/activate
./src/classification_mlp.py --data ../Representation_Probing/Representations/Grammaticality/morphology/anaphor_gender_agreement/bert_sentence_representations.pkl --name anaphor_geneder_agreement_BERT
./src/fig_classification.py --data computed/mpl_anaphor_geneder_agreement_BERT.json

./src/classification_mlp.py --data ../Representation_Probing/Representations/Grammaticality/semantics/matrix_question_npi_licensor_present/bert_sentence_representations.pkl --name matrix_question_npi_licensor_present_BERT
./src/fig_classification.py --data computed/mpl_matrix_question_npi_licensor_present_BERT.json

./src/classification_mlp.py --data ../Representation_Probing/Representations/Grammaticality/syntax/transitive/bert_sentence_representations.pkl --name transitive_BERT
./src/fig_classification.py --data computed/mpl_transitive_BERT.json
