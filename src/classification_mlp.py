#!/usr/bin/env python3

from utils import read_pickle, save_json, file_list, computed_path_generator
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import shuffle
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path
import os
from sklearn.model_selection import KFold
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

args = ArgumentParser()
args.add_argument("-d", "--data", default="data/CoLA_BERT.pkl")
# args.add_argument("--target", default="class", help="Probably `amb` or `class`. Based on Sunit's export.")
args = args.parse_args()

rep_files = file_list(args.data)
# f_name = json_name(args.data,args.target)



for model in rep_files:
    logdata = defaultdict(lambda: defaultdict(list))
    target_path = computed_path_generator(model)
    data = read_pickle(model)
    model_name = (model.split("/")[-1]).strip(".pkl")
    datset_name = (model.split("/")[-2])
    
    print("processing %s of dataset %s"%(model_name,datset_name))

    if model_name=="GPT2":
        representations = ["mean", "haddamard", "sum"]
    else:
        representations = ["pooler", "cls", "mean", "haddamard", "sum"]
 
    for mode in representations:
        print(mode)
        layer_generator = range(13) if mode != "pooler" else [0]
        for layer in layer_generator:
            for rstate in range(4):
                
                if mode == "pooler":
                    data_x = [x[mode] for x in data]
                else:
                    data_x = [x[mode][layer] for x in data]
                
                
                
                model = MLPClassifier(
                    random_state=rstate,
                    hidden_layer_sizes=(100,),
                    early_stopping=True
                )

                data_x = StandardScaler().fit_transform(data_x)
                data_y = [x["class"] for x in data]
                labels = np.unique(data_y)
                pre = preprocessing.LabelEncoder()
                pre.fit(labels)
                data_y = pre.transform(data_y)              

                cv = cross_validate(model, data_x, data_y, cv=10, n_jobs=-1, return_train_score=True)
                # print(cv.keys()) 

                # data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
                #     data_x, data_y, test_size=100, shuffle=True, random_state=0,
                # )

                # model.fit(data_x_train, data_y_train)
                score_train = cv["train_score"]
                score_test = cv["test_score"]
                score_train = np.mean(np.array(score_train))
                score_test = np.mean(np.array(score_test))
                
                print(
                    f"Layer {layer} accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}"
                )

                logdata[mode][layer].append((score_train, score_test))
    
    logfile_name = os.path.join(target_path,"mlp_" + model_name + ".json")
    print("Saving to", logfile_name)
    save_json(logfile_name, logdata)

    
