#!/usr/bin/env python3

from utils import read_json
import matplotlib.pyplot as plt
import numpy as np
import fig_utils
import scipy.stats as st

data = read_json("computed/mlp_BERT_COCO.json")


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

for k in ["mean", "haddamard", "sum", "cls"]:
    # take test scores
    ys = [[x[1] for x in data[k][str(layer)]] for layer in range(12)]
    cs = [confidence(y) for y in ys]
    yerr = [(x[1] - x[0]) / 2 for x in cs]
    plt.errorbar(
        list(range(12)),
        [np.average(y) for y in ys],
        label=k.capitalize(),
        yerr=yerr,
        **PLTARGS
    )


# special pooler handling
ys = [x[1] for x in data["pooler"]["0"]]
cs = confidence(ys)
yerr = (cs[1] - cs[0]) / 2
plt.errorbar(
    [11],
    [np.average(ys)],
    label="Pooler",
    yerr=yerr,
    ** PLTARGS
)

plt.hlines(
    0.5, 0, 11, label="MCCC",
    linestyle=":", color="tab:gray")
plt.ylabel("Dev accuracy")
plt.xlabel("Layer")

plt.tight_layout(pad=0.1)
plt.legend(ncol=2)
plt.show()
