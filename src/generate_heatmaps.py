#!/usr/bin/env python3

from utils import read_json, read_tfidf, save_json
import matplotlib.pyplot as plt
import numpy as np
import fig_utils
import seaborn as sns
from collections import defaultdict

import os
plt.style.use("seaborn")

def extract_summary(rep_loc):
    data = read_json(rep_loc)
    max_score_global = -1
    for rep_type in data.keys():
        if len(data[rep_type])==1:
            target = data[rep_type]["0"]
            test_scores = [x[1] for x in target]
            if max(test_scores)>max_score_global:
                max_score_global=max(test_scores)
        else:
            for layer in range(13):
                target = data[rep_type][str(layer)]
                test_scores = [x[1] for x in target]
                max_score = max(test_scores)
                if max_score>max_score_global:
                    max_score_global = max_score
    return max_score_global

def summary_stats(data_loc):
    dataset_name = data_loc.split("/")[-1]
    
    scores = []
    

    tf_idf = float(read_tfidf(dataset_name)) 
    scores.append(tf_idf)

    BERT = os.path.join(data_loc,"mlp_BERT.json")
    max_score = extract_summary(BERT)
    scores.append(max_score)

    GPT2 = os.path.join(data_loc,"mlp_GPT2.json")
    max_score = extract_summary(GPT2)
    scores.append(max_score)

    SBERT = os.path.join(data_loc,"mlp_SBERT.json")
    max_score = extract_summary(SBERT)
    scores.append(max_score)

    return scores



def ambiguous():
    case_scores = []
    ambiguous_location = os.path.join(os.getcwd(),os.path.join("computed","Ambiguity"))
    
    COCO = os.path.join(ambiguous_location,"COCO")
    scores = summary_stats(COCO)
    case_scores.append(scores)
    
    EMMT = os.path.join(ambiguous_location,"EMMT")
    scores = summary_stats(EMMT)
    case_scores.append(scores)
    
    return case_scores
    
def CoLA():
    case_scores = []
    ambiguous_location = os.path.join(os.getcwd(),os.path.join("computed","Grammaticality"))
    
    CoLA = os.path.join(ambiguous_location,"CoLA")
    scores = summary_stats(CoLA)
    case_scores.append(scores)
    
    
    return case_scores
    

Dataset_names = ["COCO","EMMT","CoLA"]
All_data = []
amb_data_scores = ambiguous() 
All_data = All_data + amb_data_scores
CoLA_data_scores = CoLA() 
All_data = All_data + CoLA_data_scores
Dataset = np.array(All_data)
print(Dataset.shape)
text = np.repeat(np.array(["T","B","G","S"]),len(Dataset_names))
print(text.shape)
# fig, ax = plt.subplots()
#   # drawing heatmap on current axes
# ax = sns.heatmap(Dataset, annot=text, fmt="")
# plt.savefig("Heatmap")