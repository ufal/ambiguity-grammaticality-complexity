#!/usr/bin/env python3

from utils import read_json
import matplotlib.pyplot as plt
import numpy as np
import fig_utils
import scipy.stats as st
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("-d", "--data", default="computed/mlp_CoLA_BERT.json")
args.add_argument("--tfidf", "--tf-idf", type=float, default=None)
args = args.parse_args()

data = read_json(args.data)

def confidence(vals):
    return st.t.interval(
        alpha=0.95,
        df=len(vals) - 1,
        loc=np.mean(vals),
        scale=st.sem(vals)
    )


PLTARGS = dict(
    capsize=3, capthick=2,
    ms=10, marker=".",
    elinewidth=1
)

for k in ["Mean", "Haddamard", "Sum", "CLS"]:
    # take test scores
    ys = [[x[1] for x in data[k.lower()][str(layer)]] for layer in range(12)]
    cs = [confidence(y) for y in ys]
    yerr = [(x[1] - x[0]) / 2 for x in cs]
    plt.errorbar(
        list(range(12)),
        [np.average(y) for y in ys],
        label=k,
        yerr=yerr,
        **PLTARGS
    )


# special pooler handling
ys = [x[1] for x in data["pooler"]["0"]]
cs = confidence(ys)
yerr = (cs[1] - cs[0]) / 2
plt.errorbar(
    [10.5],
    [np.average(ys)],
    label="Pooler",
    yerr=yerr,
    ** PLTARGS
)

plt.hlines(
    0.5, 0, 11, label="MCCC",
    linestyle=":", color="tab:gray"
)

if args.tfidf is not None:
    plt.hlines(
        args.tfidf, 0, 11, label="TF-IDF",
        linestyle="-.", color="tab:gray"
    )

plt.ylabel("Dev accuracy")
plt.xlabel("Layer")

plt.tight_layout(pad=0.1)
plt.legend(ncol=2)
plt.savefig(args.data.replace(".json", ".pdf"))
plt.show()
