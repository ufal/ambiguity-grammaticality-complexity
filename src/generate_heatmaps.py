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
    
    max_scores = []    

    tf_idf = float(read_tfidf(dataset_name)) 
    max_scores.append(tf_idf)

    BERT = os.path.join(data_loc,"mlp_BERT.json")
    max_score = extract_summary(BERT)
    max_scores.append(max_score)
    
    GPT2 = os.path.join(data_loc,"mlp_GPT2.json")
    max_score = extract_summary(GPT2)
    max_scores.append(max_score)
   
    SBERT = os.path.join(data_loc,"mlp_SBERT.json")
    max_score = extract_summary(SBERT)
    max_scores.append(max_score)
    
    return max_scores



def ambiguous():
    case_scores_max = []
    
    ambiguous_location = os.path.join(os.getcwd(),os.path.join("computed","Ambiguity"))
    
    COCO = os.path.join(ambiguous_location,"COCO")
    max_scores  = summary_stats(COCO)
    case_scores_max.append(max_scores)
    
    
    EMMT = os.path.join(ambiguous_location,"EMMT")
    max_scores = summary_stats(EMMT)
    case_scores_max.append(max_scores)
        
    return case_scores_max
    
def CoLA():
    case_scores_max = []

    data_location = os.path.join(os.getcwd(),os.path.join("computed","Grammaticality"))
    
    CoLA = os.path.join(data_location,"CoLA")
    max_scores = summary_stats(CoLA)
    case_scores_max.append(max_scores)

    return case_scores_max
    

Dataset_names = ["COCO","EMMT","CoLA"]
All_data_max = []

amb_data_scores_max = ambiguous() 
All_data_max = All_data_max + amb_data_scores_max

CoLA_data_scores_max = CoLA() 
All_data_max = All_data_max + CoLA_data_scores_max


Dataset_max = np.array(All_data_max)
x_axis_labels = ["TF-IDF","BERT","GPT2","SBERT"] # labels for x-axis
y_axis_labels = Dataset_names # labels for y-axis

fig, ax = plt.subplots()
  # drawing heatmap on current axes
ax = sns.heatmap(Dataset_max, xticklabels=x_axis_labels, yticklabels=y_axis_labels, fmt="")
plt.title("Heatmap: Max")
plt.savefig("computed/Heatmap_max.pdf")

plt.clf()
plt.close()

