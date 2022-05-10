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
args.add_argument("-d1", "--data-1", default="computed/mlp_BERT_COCO.json")
args.add_argument("-d2", "--data-2", default="computed/mlp_BERT_COMPEN.json")
args = args.parse_args()

name = ((args.data_1).split("/")[-1]).split("_")[1].replace(".json","")


tf_idf_1 = float(read_tfidf_neural("COCO"))
tf_idf_2 = float(read_tfidf_neural("complexity_english"))

PLTARGS = dict(
    capsize=3, capthick=2,
    ms=10, marker=".",
    elinewidth=1
)
data1 = read_json(args.data_1)
data2 = read_json(args.data_2)

# plt.figure(figsize=(4.5, 4))
plt.rcParams["figure.figsize"] = (4.5,4)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

if name=="GPT2":
    reps =  ["Mean", "Haddamard"]
else:
    reps =  ["Mean", "Haddamard", "CLS"]


for data_i, (data, tf_idf, ax, title) in enumerate(zip(
        [data1, data2], [tf_idf_1, tf_idf_2], [ax1, ax2], ["Ambiguity", "Complexity"]
    )):
    for k in reps:
        # take test scores
        ys = [
            [x[1] for x in data[k.lower()][str(layer)]]
            for layer in range(13)
        ]
        cs = [confidence(y) for y in ys]
        yerr = [(x[1] - x[0]) / 2 for x in cs]
        ax.errorbar(
            list(range(13)),
            [np.average(y) for y in ys],
            # hotfix
            label=k.replace("Haddamard", "Hadamard"),
            yerr=yerr,
            **PLTARGS
        )
    if name!="GPT2":
        # special pooler handling
        ys = [x[1] for x in data["pooler"]["0"]]
        cs = confidence(ys)
        yerr = (cs[1] - cs[0]) / 2
        ax.errorbar(
            [12.5],
            [np.average(ys)],
            label="Pooler",
            yerr=yerr,
            ** PLTARGS
        )


    ax.hlines(
        tf_idf, 0, 12.5, label="TF-IDF",
        linestyle=":", color="tab:gray"
    )
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel(f"{title} acc.")
    
    if data_i == 0:
        # only the first one because there's space
        ax.legend(ncol=3, loc="upper center")
        ax.get_xaxis().set_visible(False)

# implicitply the last one
plt.xlabel("Layer")

plt.tight_layout(pad=0.1)

plt.savefig("computed/amb_comp_double.pdf")
plt.show()