#!/usr/bin/env python3

from utils import read_pickle, save_json, json_name
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import shuffle
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path

args = ArgumentParser()
args.add_argument("-d", "--data", default="data/CoLA_BERT.pkl")
args.add_argument("--target", default="class", help="Probably `amb` or `class`. Based on Sunit's export.")
args = args.parse_args()
f_name = json_name(args.data,args.target)

data = read_pickle(args.data)

logdata = defaultdict(lambda: defaultdict(list))

model = f_name.split("_")[0]
if model=="SBERT":
    print("sbert-embedding")
    for rstate in range(8):
                model = MLPClassifier(
                    random_state=rstate,
                    hidden_layer_sizes=(100,),
                    early_stopping=True
                )

                data_x = [x['embedding'] for x in data]  
                data_x = StandardScaler().fit_transform(data_x)
                data_y = [x[args.target] for x in data]

                data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
                    data_x, data_y, test_size=100, shuffle=True, random_state=0,
                )

                model.fit(data_x_train, data_y_train)
                score_train = model.score(data_x_train, data_y_train)
                score_test = model.score(data_x_test, data_y_test)

                print(
                    f"SBERT Representation accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}"
                )

                logdata['embedding'][0].append((score_train, score_test))

    logfile_name = "computed/mlp_" + f_name + ".json"
    print("Saving to", logfile_name)
    save_json(logfile_name, logdata)

else:
    if model=="BERT":
        rep_list = ["pooler", "cls", "mean", "haddamard", "sum"]
    elif model=="GPT":
        rep_list = ["mean", "haddamard", "sum"]

    for mode in rep_list:
        print(mode)

        layer_generator = range(12) if mode != "pooler" else [0]

        for layer in layer_generator:
            for rstate in range(8):
                model = MLPClassifier(
                    random_state=rstate,
                    hidden_layer_sizes=(100,),
                    early_stopping=True
                )

                if mode == "pooler":
                    data_x = [x[mode] for x in data]
                else:
                    data_x = [x[mode][layer] for x in data]

                data_x = StandardScaler().fit_transform(data_x)
                data_y = [x[args.target] for x in data]

                data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
                    data_x, data_y, test_size=100, shuffle=True, random_state=0,
                )

                model.fit(data_x_train, data_y_train)
                score_train = model.score(data_x_train, data_y_train)
                score_test = model.score(data_x_test, data_y_test)

                print(
                    f"Layer {layer} accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}"
                )

                logdata[mode][layer].append((score_train, score_test))

    logfile_name = "computed/mlp_" + f_name + ".json"
    print("Saving to", logfile_name)
    save_json(logfile_name, logdata)
