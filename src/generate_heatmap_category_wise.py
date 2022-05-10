#!/usr/bin/env python3

raise Exception("This script is deprecated")

from utils import read_json, read_tfidf
import matplotlib.pyplot as plt
import numpy as np
import fig_utils
from argparse import ArgumentParser
import os
from utils import computed_path_generator
from pathlib import Path

args = ArgumentParser()
args.add_argument("-d", "--data", default="computed/Ambiguity/COCO")
args = args.parse_args()
data_path = args.data

case_name = data_path.split("/")[-1]
if len(case_name.split())==0:
    case_name = data_path.split("/")[-2]

tfidf = float(read_tfidf(case_name))

File_List = ["mlp_BERT.json","mlp_GPT2.json","mlp_SBERT.json"]

for representation_type in File_List:
    data = read_json(os.path.join(data_path,representation_type))
    if "GPT2" in representation_type:
        rep_list = ["Mean", "Haddamard", "Sum"]
        imdata = np.zeros((3 + 1, 13))
    else:
        rep_list = ["Mean", "Haddamard", "Sum", "CLS"]
        imdata = np.zeros((4 + 1, 13))
    
    for k_i,k in enumerate(rep_list):
        # take test scores
        ys = [[x[1] for x in data[k.lower()][str(layer)]] for layer in range(13)]
        imdata[k_i] = [np.average(y) for y in ys]


    # set bottom row
    imdata[-1][:4] = 0.5
    if tfidf is not None:
        imdata[-1][4:8] = tfidf
    else:
        imdata[-1][4:8] = 0.5
    # imdata[-1][8:] = np.average([x[1] for x in data["pooler"]["0"]])

    plt.figure(figsize=(1.5, 0.6))
    plt.imshow(imdata, aspect="auto")
    plt.axis('off')
    plt.tight_layout(pad=0.1)

    file_name = "sample_COCO.pdf"
    # save_path = args.data.replace(".json", ".pdf")
    print("Saving to", file_name)

    plt.savefig(file_name)
    plt.show()
