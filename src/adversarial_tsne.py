#!/usr/bin/env python3

from utils import read_pickle, save_pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import fig_utils
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("-d", "--data", default="data/ambiguity_COCO_BERT.pkl")
args.add_argument("-c", "--compute", action="store_true")
args = args.parse_args()
data = read_pickle(args.data)

if args.compute:
    classes = list({x["class"] for x in data})
    print("Classes:", classes)

    data_a = np.array([x["cls"][1] for x in data if x["class"] == classes[0]])
    data_b = np.array([x["cls"][1] for x in data if x["class"] == classes[1]])
    model = TSNE(
        n_components=2, init="pca",
        learning_rate="auto", random_state=1
    )
    data = model.fit_transform(np.concatenate((data_a, data_b)))
    orig_a = data[:len(data_a)]
    orig_b = data[len(data_a):]

    print("Original data:", len(data_a), len(data_b))
    dists = euclidean_distances(data_a, data_b)
    print("Distances:", dists.shape)
    max_i = np.max(np.argmax(dists, axis=0))
    max_j = np.argmax(np.argmax(dists, axis=0))
    print("Highest:", max_i, max_j, dists[max_i, max_j])

    CLUSTER_SIZE = 231
    seed_a = data_a[max_i]
    seed_b = data_b[max_j]

    dists_a_1 = euclidean_distances([seed_b], data_a)[0]
    cluster_a_1 = dists_a_1.argsort()[-CLUSTER_SIZE // 2:]
    dists_a_2 = euclidean_distances([seed_a], data_a)[0]
    cluster_a_2 = dists_a_2.argsort()[:CLUSTER_SIZE // 2]
    cluster_a = np.concatenate((cluster_a_1, cluster_a_2))
    print(data_a[cluster_a].shape)

    dists_b_1 = euclidean_distances([seed_a], data_b)[0]
    cluster_b_1 = dists_a_1.argsort()[-CLUSTER_SIZE // 2:]
    dists_b_2 = euclidean_distances([seed_b], data_b)[0]
    cluster_b_2 = dists_a_2.argsort()[:CLUSTER_SIZE // 2]
    cluster_b = np.concatenate((cluster_b_1, cluster_b_2))
    print(data_b[cluster_b].shape)

    model = TSNE(
        n_components=2, init="pca",
        learning_rate="auto", random_state=1
    )
    data_tsne = model.fit_transform(np.concatenate(
        (data_a[cluster_a], data_b[cluster_b])))
    assert len(data_tsne) == 2 * CLUSTER_SIZE
    adv_a = data_tsne[:CLUSTER_SIZE]
    adv_b = data_tsne[CLUSTER_SIZE:]

    save_pickle(
        "computed/tsne_force.pkl",
        (((orig_a, orig_b), (adv_a, adv_b)))
    )
else:
    data = read_pickle("computed/tsne_force.pkl")
    orig_a, orig_b = data[0]
    adv_a, adv_b = data[1]


plt.figure(figsize=(4.5, 2.7))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)


def norm_data(data_sub):
    max_x = max([x for x, y in data_sub])
    min_x = min([x for x, y in data_sub])
    max_y = max([y for x, y in data_sub])
    min_y = min([y for x, y in data_sub])
    return [
        (
            (x - min_x) / (max_x - min_x),
            (y - min_y) / (max_y - min_y)
        )
        for x, y in data_sub
    ]


for ax, (d1, d2), text in zip([ax1, ax2], [(orig_a, orig_b), (adv_a, adv_b)], ["461A + 461U", "231A + 231U"]):
    d1 = norm_data(d1)
    d2 = norm_data(d2)
    ax.scatter(
        [x[0] for x in d1],
        [x[1] for x in d1],
        s=10,
    )
    ax.scatter(
        [x[0] for x in d2],
        [x[1] for x in d2],
        s=10,
    )

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.09, 1.2)
    ax.text(0.005, 1.05, text, ha="left")

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout(rect=[0, 0, 0.99, 1], pad=0)
plt.savefig("computed/tsne_force.pdf")
plt.show()
