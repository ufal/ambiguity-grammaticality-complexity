#!/usr/bin/env python3

from utils import read_json, read_tfidf_neural
import matplotlib.pyplot as plt
import numpy as np
import fig_utils
import scipy.stats as st
from argparse import ArgumentParser


def confidence(vals):
    return st.t.interval(
        alpha=0.95,
        df=len(vals) - 1,
        loc=np.mean(vals),
        scale=st.sem(vals)
    )


args = ArgumentParser()
args.add_argument("-d", "--data", default="computed/Grammaticality/aggregation_max.json")
args = args.parse_args()

data = read_json(args.data)
# sort data by decreasing length
data.sort(key=lambda x: len(x), reverse=True)
data = [data[i] for i in [1,0,2,3]]

PLTARGS = dict(
    capsize=3, capthick=2,
    ms=10, marker=".",
    elinewidth=1
)

fig = plt.figure(figsize=(4.5, 2.8))

for data_i, data_v in enumerate(data):
    ax = plt.subplot(5, 1, data_i + 1)
    img = np.zeros((max([len(x) for x in data]), 3))

    for task_i, task in enumerate(data_v):
        img[task_i][0] = task["TFIDF"]
        img[task_i][1] = task["BERT"]
        img[task_i][2] = task["GPT"]
    
    
    # mask zeroes
    img = np.ma.masked_where(img == 0, img)
    img_p = ax.imshow(
        img.T, vmin=min(img[img > 0]), vmax=1, aspect="auto",
        cmap="cividis"
    )

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.text(
        img.shape[0]-1, 1.5,
        task["condition"].replace("_", " ").capitalize(),
        ha="right"
    )

    for text_i, text in enumerate(["TF-IDF", "BERT", "GPT-2"]):
        ax.text(
            -1, text_i,
            text, ha="right", va="center",
            fontsize=9,
        )
    

cax = plt.axes([0.07, 0.1, 0.9, 0.05])
plt.colorbar(img_p, orientation="horizontal", cax=cax)

plt.subplots_adjust(
    top=0.98, right=0.98, bottom=-0.04,
    hspace=0.05, wspace=0
)

plt.savefig("computed/grammaticality_aggregation.pdf")
plt.show()
