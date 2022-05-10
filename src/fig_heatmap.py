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
args.add_argument("-d", "--data", default="computed/Grammaticality/morphology/determiner_noun_agreement_irregular_1/mlp_BERT.json")
args = args.parse_args()

# data = read_json(args.data)
data = {k:np.random.rand(3) for k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

# tf_idf = float(read_tfidf_neural("determiner_noun_agreement_irregular_1"))

PLTARGS = dict(
    capsize=3, capthick=2,
    ms=10, marker=".",
    elinewidth=1
)

plt.figure(figsize=(4.5, 2.97))

img = np.zeros((len(data), 3))

for task_i, (task, task_v) in enumerate(data.items()):
    img[task_i][0] = task_v[0]
    img[task_i][1] = task_v[1]
    img[task_i][2] = task_v[2]

plt.imshow(img)

for task_i, (task, task_v) in enumerate(data.items()):
    plt.text(-1, task_i, task)

plt.tight_layout(
    rect=[0, 0, 1, 1.01],
    pad=0.1
)

# plt.savefig("computed/blimp_det_noun.pdf")
plt.show()