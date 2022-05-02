#!/usr/bin/env python3

from utils import read_json, read_tfidf
import matplotlib.pyplot as plt
import numpy as np
import fig_utils
from argparse import ArgumentParser
import os
from utils import computed_path_generator
from pathlib import Path

args = ArgumentParser()
args.add_argument("-d", "--data", default="computed/mlp_CoLA_BERT.json")
args.add_argument("-t", "--task", default="CoLA")
args = args.parse_args()

tfidf = float(read_tfidf(args.task))

data = read_json(args.data)

tfidf_extra = 1 if tfidf is not None else 0

if "GPT2" in args.data:
    rep_list = ["Mean", "Haddamard"]
    imdata = np.zeros((2 + tfidf_extra, 12))
else:
    rep_list = ["Mean", "CLS", "Haddamard"]
    imdata = np.zeros((3 + tfidf_extra, 12))

for k_i, k in enumerate(rep_list):
    # take test scores
    print(data.keys())
    print(data[k.lower()].keys())
    ys = [
        [x[1] for x in data[k.lower()][str(layer)]]
        for layer in range(12)
    ]
    imdata[k_i] = [np.average(y) for y in ys]

# set bottom row
imdata[-1][:4] = 0.5
if tfidf is not None:
    imdata[-1][4:8] = tfidf
else:
    imdata[-1][4:8] = 0.5
imdata[-1][8:] = np.average([x[1] for x in data["pooler"]["0"]])


plt.figure(figsize=(1.5, 0.6))
plt.imshow(imdata, aspect="auto")
plt.axis('off')
plt.tight_layout(pad=0.1)

save_path = args.data.replace(".json", ".pdf")
print("Saving to", save_path)

plt.savefig(save_path)
plt.show()
