#!/usr/bin/env python3

from utils import read_pickle
import os
from argparse import ArgumentParser
from sklearn import preprocessing
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

args = ArgumentParser()
args.add_argument("-d", "--data", default="multilingual/computed/Ambiguity/COCO")
args.add_argument("-g", "--graph_loc", default="multilingual/graphs/T-SNE")
args.add_argument("-m", "--mode", default="Ambiguity")
args = args.parse_args()


t_sne_loc = os.path.join(args.graph_loc,"T-SNE")

if not os.path.exists(t_sne_loc):   
    os.mkdir(t_sne_loc)
case_loc = os.path.join(t_sne_loc,args.mode)
if not os.path.exists(case_loc):   
    os.mkdir(case_loc)
data_loc = os.path.join(case_loc,(args.data).split("/")[-1])
if not os.path.exists(data_loc):   
    os.mkdir(data_loc)

# models = ["BERT","GPT2","SBERT"]

models = ["BERT","GPT2"]


for model in models:
    rep_name = os.path.join(args.data,model+".pkl")
    data = read_pickle(rep_name)

    if model=="GPT2":
        representations = ["mean", "haddamard", "sum"]
    else:
        representations = ["mean", "haddamard", "sum", "pooler", "cls"]
    
    for mode in representations:
        print(mode)
        layer_generator = range(13) if mode != "pooler" else [0]
        for layer in layer_generator:
            if mode == "pooler":
                data_x = [x[mode] for x in data]
            else:
                data_x = [x[mode][layer] for x in data]
            
            data_y = [x["class"] for x in data]

            # labels = np.unique(data_y)
            # pre = preprocessing.LabelEncoder()
            # pre.fit(labels)
            # data_y = pre.transform(data_y)  
            try:
                tsne = TSNE(n_components=2, verbose=1, random_state=123)
                z = tsne.fit_transform(data_x)
                df = pd.DataFrame()
                df["y"] = data_y
                df["component-1"] = z[:,0]
                df["component-2"] = z[:,1]
                # title = "Type: "+(args.mode).upper() +"; Dataset: "+(args.data).split("/")[-1]+"; Model: "+ model+"; Mode: "+mode
                name_fig = (args.mode).upper() +"_"+ model+"_"+mode
                if mode != "pooler":
                    # title = title + " Layer: " + str(layer)            
                    name_fig = name_fig+"_layer"+str(layer)           
                path_loc = os.path.join(data_loc,name_fig+".csv")
                df.to_csv(path_or_buf=path_loc,index=False)
                # sns.scatterplot(x="component-1", y="component-2", hue=df.y.tolist(),
                #     palette=sns.color_palette("hls", 2),
                #     data=df).set(title=title)
                # plt.savefig(os.path.join(data_loc,name_fig+".jpg"))
                # plt.clf()
                # plt.close()
            except ValueError:
                print("problem with %s %s %s"%(model,mode,str(layer)))
         