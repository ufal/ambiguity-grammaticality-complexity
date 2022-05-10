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
args.add_argument("-d", "--data", default="computed/Complexity/complexity_english/mlp_BERT.json")
args = args.parse_args()

tf_idf = float(read_tfidf_neural("complexity_english"))

PLTARGS = dict(
    capsize=3, capthick=2,
    ms=10, marker=".",
    elinewidth=1
)
data = read_json(args.data)

plt.figure(figsize=(4.5, 4))
ax = plt.gca()

for k in ["Mean", "Haddamard", "CLS"]:
    print(k)
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
ax.set_ylabel(f"Accuracy")

ax.legend(ncol=3, loc="upper center")
ax.get_xaxis().set_visible(False)

plt.xlabel("Layer")

plt.tight_layout(pad=0.1)

plt.savefig("computed/blimp_det_noun.pdf")
plt.show()