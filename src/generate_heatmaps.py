#!/usr/bin/env python3

raise Exception("This script is deprecated")

from utils import read_json, read_tfidf, read_tfidf_neural, save_json
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
    layer_score = -1
    rep = ''
    for rep_type in data.keys():
        if len(data[rep_type])==1:
            target = data[rep_type]["0"]
            test_scores = [x[1] for x in target]
            if max(test_scores)>max_score_global:
                max_score_global=max(test_scores)
                layer_score=13            
                rep=rep_type
        else:
            for layer in range(13):
                target = data[rep_type][str(layer)]
                test_scores = [x[1] for x in target]
                max_score = max(test_scores)
                if max_score>max_score_global:
                    max_score_global = max_score
                    layer_score=layer
                    rep=rep_type
    
   
    return max_score_global,layer_score,rep

def summary_stats(data_loc):
    dataset_name = data_loc.split("/")[-1]
    
    max_scores = []    
    layer_scores = []
    representation_types = []

    tf_idf = float(read_tfidf(dataset_name)) 
    max_scores.append(tf_idf)

    tf_idf_neural = float(read_tfidf_neural(dataset_name)) 
    max_scores.append(tf_idf_neural)


    BERT = os.path.join(data_loc,"mlp_BERT.json")
    max_score,layer_score,rep = extract_summary(BERT)
    max_scores.append(max_score)
    layer_scores.append(layer_score)
    representation_types.append(rep)
    
    GPT2 = os.path.join(data_loc,"mlp_GPT2.json")
    max_score,layer_score,rep = extract_summary(GPT2)
    max_scores.append(max_score)
    layer_scores.append(layer_score)
    representation_types.append(rep)
   
    SBERT = os.path.join(data_loc,"mlp_SBERT.json")
    max_score,layer_score,rep = extract_summary(SBERT)
    max_scores.append(max_score)
    layer_scores.append(layer_score)
    representation_types.append(rep)

    return max_scores, layer_scores, representation_types

def check_BliMP(loc):
    flag = 0
    if os.path.exists(os.path.join(loc,"mlp_BERT.json")):
        flag+=1
    if os.path.exists(os.path.join(loc,"mlp_GPT2.json")):
        flag+=1
    if os.path.exists(os.path.join(loc,"mlp_SBERT.json")):
        flag+=1
    if flag==3:
        return True
    else: 
        return False

def BlIMP_morphology():
    scores_agg = []
    names = []
    layers_agg = []
    rep_type_agg = []

    data_location = os.path.join(os.getcwd(),os.path.join("computed",os.path.join("Grammaticality","morphology")))
    case_scores = []
    count = 0
    f_ = open(os.path.join(os.getcwd(),os.path.join("computed","morphology_task_names")),"w")
    f_.close()
    for folder in os.listdir(data_location):
        data = os.path.join(data_location,folder)
        if check_BliMP(data):
            count+=1
            task_name = "Task_"+str(count)
            f_ = open(os.path.join(os.getcwd(),os.path.join("computed","morphology_task_names")),"a")
            print("%s\t%s"%(folder,task_name),file=f_)
            f_.close()
            names.append(task_name)
            scores, layer_scores, rep = summary_stats(data)
            scores_agg.append(scores)
            layers_agg.append(layer_scores)
            rep_type_agg.append(rep)
    return scores_agg,names,layers_agg,rep_type_agg

def BlIMP_syntax():
    scores_agg = []
    names = []
    layers_agg = []
    rep_type_agg = []

    data_location = os.path.join(os.getcwd(),os.path.join("computed",os.path.join("Grammaticality","syntax")))
    case_scores = []
    count = 0
    f_ = open(os.path.join(os.getcwd(),os.path.join("computed","syntax_task_names")),"w")
    f_.close()
    for folder in os.listdir(data_location):
        data = os.path.join(data_location,folder)
        if check_BliMP(data):
            count+=1
            task_name = "Task_"+str(count)
            f_ = open(os.path.join(os.getcwd(),os.path.join("computed","syntax_task_names")),"a")
            print("%s\t%s"%(folder,task_name),file=f_)
            f_.close()
            names.append(task_name)
            scores, layer_scores, rep = summary_stats(data)
            scores_agg.append(scores)
            layers_agg.append(layer_scores)
            rep_type_agg.append(rep)
    return scores_agg,names,layers_agg,rep_type_agg

def BlIMP_semantics():
    scores_agg = []
    names = []
    layers_agg = []
    rep_type_agg = []

    data_location = os.path.join(os.getcwd(),os.path.join("computed",os.path.join("Grammaticality","semantics")))
    case_scores = []
    count = 0
    f_ = open(os.path.join(os.getcwd(),os.path.join("computed","semantics_task_names")),"w")
    f_.close()
    for folder in os.listdir(data_location):
        data = os.path.join(data_location,folder)
        if check_BliMP(data):
            count+=1
            task_name = "Task_"+str(count)
            f_ = open(os.path.join(os.getcwd(),os.path.join("computed","semantics_task_names")),"a")
            print("%s\t%s"%(folder,task_name),file=f_)
            f_.close()
            names.append(task_name)
            scores, layer_scores, rep = summary_stats(data)
            scores_agg.append(scores)
            layers_agg.append(layer_scores)
            rep_type_agg.append(rep)
    return scores_agg,names,layers_agg,rep_type_agg

def BlIMP_syntax_semantics():
    scores_agg = []
    names = []
    layers_agg = []
    rep_type_agg = []

    data_location = os.path.join(os.getcwd(),os.path.join("computed",os.path.join("Grammaticality","syntax_semantics")))
    case_scores = []
    count = 0
    f_ = open(os.path.join(os.getcwd(),os.path.join("computed","syntax_semantics_task_names")),"w")
    f_.close()
    for folder in os.listdir(data_location):
        data = os.path.join(data_location,folder)
        if check_BliMP(data):
            count+=1
            task_name = "Task_"+str(count)
            f_ = open(os.path.join(os.getcwd(),os.path.join("computed","syntax_semantics_task_names")),"a")
            print("%s\t%s"%(folder,task_name),file=f_)
            f_.close()
            names.append(task_name)
            scores, layer_scores, rep = summary_stats(data)
            scores_agg.append(scores)
            layers_agg.append(layer_scores)
            rep_type_agg.append(rep)
    return scores_agg,names,layers_agg,rep_type_agg
    
       

def ambiguous_CoLA():
    scores_agg = []
    names = []
    layers_agg = []
    rep_type_agg = []

    
    
    Datasets = ["COCO","EMMT"]
    ambiguous_location = os.path.join(os.getcwd(),os.path.join("computed","Ambiguity"))
    case_scores = []

    for dataset in Datasets:
        data = os.path.join(ambiguous_location,dataset)    
        scores, layer_scores, rep = summary_stats(data)
        scores_agg.append(scores)
        layers_agg.append(layer_scores)
        rep_type_agg.append(rep)
    
    CoLA_location = os.path.join(os.getcwd(),os.path.join("computed","Grammaticality"))
    data = os.path.join(CoLA_location,"CoLA")    
    scores, layer_scores, rep = summary_stats(data)
    scores_agg.append(scores)
    layers_agg.append(layer_scores)
    rep_type_agg.append(rep)
    
    names = ["COCO","EMMT","CoLA"]    
    x_axis_labels = ["TF-IDF","BERT","GPT2","SBERT"]
    fig, ax = plt.subplots()
    ax = sns.heatmap(scores_agg, xticklabels=x_axis_labels,  yticklabels=names, fmt="")

    plt.title("Heatmap: Ambiguous and CoLA")
    plt.tight_layout(pad=0.1)
    plt.savefig("computed/Heatmap_Ambiguous_CoLA.pdf")

    plt.clf()
    plt.close()

    x_axis_labels = ["BERT","GPT2","SBERT"] 
    fig, ax = plt.subplots()
    ax = sns.heatmap(layers_agg, annot=rep_type_agg, xticklabels=x_axis_labels,  yticklabels=names, fmt="")

    plt.title("Heatmap: Ambiguous and CoLA layers")
    plt.tight_layout(pad=0.1)
    plt.savefig("computed/Heatmap_Ambiguous_CoLA_layers.pdf")

    plt.clf()
    plt.close()
        

def BLiMP_Plotter(condition):
    if condition=="morphology":
        func = BlIMP_morphology()
    if condition=="syntax":
        func = BlIMP_syntax()
    if condition=="semantics":
        func = BlIMP_semantics()
    if condition=="syntax_semantics":
        func = BlIMP_syntax_semantics()

    scores_agg, names, layer_agg, rep_agg = func

    x_axis_labels = ["TF-IDF(logistic)","TF-IDF(neural)","BERT","GPT2","SBERT"]
    fig, ax = plt.subplots()
    ax = sns.heatmap(scores_agg, xticklabels=x_axis_labels,  yticklabels=names, fmt="")

    plt.title("Heatmap: BLiMP("+condition+")")
    plt.tight_layout(pad=0.1)
    plt.savefig("computed/Heatmap_"+condition+".pdf")

    plt.clf()
    plt.close()



    x_axis_labels = ["BERT","GPT2","SBERT"] 
    fig, ax = plt.subplots()
    ax = sns.heatmap(layer_agg, annot=rep_agg, xticklabels=x_axis_labels,  yticklabels=names, fmt="")

    plt.title("Heatmap: BLiMP("+condition+") layers")
    plt.tight_layout(pad=0.1)
    plt.savefig("computed/Heatmap_"+condition+"_layers.pdf")

    plt.clf()
    plt.close()


ambiguous_CoLA()


BLiMP_Plotter("morphology")
BLiMP_Plotter("syntax")
BLiMP_Plotter("semantics")
BLiMP_Plotter("syntax_semantics")

def complexity():
    scores_agg = []
    names = []
    layers_agg = []
    rep_type_agg = []

    
    
    Datasets = ["complexity_english","complexity_italian"]
    complexity_location = os.path.join(os.getcwd(),os.path.join("computed","Complexity"))
    case_scores = []

    for dataset in Datasets:
        data = os.path.join(complexity_location,dataset)    
        scores, layer_scores, rep = summary_stats(data)
        scores_agg.append(scores)
        layers_agg.append(layer_scores)
        rep_type_agg.append(rep)
    
    
    names = ["English","Italian"]    
    x_axis_labels = ["TF-IDF","BERT","GPT2","SBERT"]
    fig, ax = plt.subplots()
    ax = sns.heatmap(scores_agg, xticklabels=x_axis_labels,  yticklabels=names, fmt="")

    plt.title("Heatmap: Complexity")
    plt.tight_layout(pad=0.1)
    plt.savefig("computed/Complexity.pdf")

    plt.clf()
    plt.close()

    x_axis_labels = ["BERT","GPT2","SBERT"] 
    fig, ax = plt.subplots()
    ax = sns.heatmap(layers_agg, annot=rep_type_agg, xticklabels=x_axis_labels,  yticklabels=names, fmt="")

    plt.title("Heatmap: Complexity layers")
    plt.tight_layout(pad=0.1)
    plt.savefig("computed/Heatmap_Complexity_layers.pdf")

    plt.clf()
    plt.close()

complexity()