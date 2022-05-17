#!/usr/bin/env python3

from utils import read_json, read_tfidf_neural
import matplotlib.pyplot as plt
import numpy as np
import fig_utils
import scipy.stats as st
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("-d1", "--data-1", default="computed/tsne_vectors/T-SNE/Complexity/complexity_english/COMPLEXITY_BERT_cls_layer2.csv")
args.add_argument("-d2", "--data-2", default="computed/tsne_vectors/T-SNE/Complexity/complexity_english/COMPLEXITY_BERT_mean_layer12.csv")
args.add_argument("-d3", "--data-3", default="computed/tsne_vectors/T-SNE/Ambiguity/COCO/AMBIGUITY_BERT_cls_layer2.csv")
args = args.parse_args()

def parse_data(path, classes):
    with open(path, "r") as f:
        data = [x.strip().split(",") for x in f.readlines()[1:]]
        data = [(x[0], float(x[1]), float(x[2])) for x in data]
        return [
            [x[1:3] for x in data if x[0] == classes[0]],
            [x[1:3] for x in data if x[0] == classes[1]]
        ]

data1 = parse_data(args.data_1, ["S", "C"])
data2 = parse_data(args.data_2, ["S", "C"])
data3 = parse_data(args.data_3, ["A", "U"])

def norm_data(data):
    def norm_subdata(data_sub):
        max_x = max([x for x, y in data_sub])
        min_x = min([x for x, y in data_sub])
        max_y = max([y for x, y in data_sub])
        min_y = min([y for x, y in data_sub])
        return [
            (
                (x-min_x)/(max_x-min_x),
                (y-min_y)/(max_y-min_y)
            )
            for x, y in data_sub
        ]

    data[0] = norm_subdata(data[0])
    data[1] = norm_subdata(data[1])
    return data

plt.figure(figsize=(4.5, 4.6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

for data_i, (data, ax, text_title) in enumerate(zip(
    [data1, data2, data3],
    [ax1, ax2, ax3],
    [
        "Complexity, [CLS] layer 2 | Cluster | 86% (+38%)", "Complexity, mean layer 12 | No cluster | 83% (+35%)", "Ambiguity, [CLS] layer 2 | No cluster | 75% (-9.2%)"
    ],
)):
    data = norm_data(data)

    ax.scatter(
        [x[0] for x in data[0]],
        [x[1] for x in data[0]],
        s=15, marker=".",
        color="black",
    )
    ax.scatter(
        [x[0] for x in data[1]],
        [x[1] for x in data[1]],
        s=15, marker=".",
        color="salmon",
    )
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.09, 1.2)
    ax.text(0.005, 1.05, text_title, ha="left")

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout(rect=[0,0,1.07,1], pad=0)
plt.savefig("computed/tsne_triple.pdf")
plt.show()
