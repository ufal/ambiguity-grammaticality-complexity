#!/usr/bin/env python3

from utils import read_pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import fig_utils
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("-d", "--data", default="data/ambiguity_COCO_BERT.pkl")
args = args.parse_args()

data = read_pickle(args.data)

data_a = np.array([x["cls"][2] for x in data if x["class"] == "A"])
data_b = np.array([x["cls"][2] for x in data if x["class"] == "U"])
print("Original data:", len(data_a), len(data_b))
dists = euclidean_distances(data_a, data_b)
print("Distances:", dists.shape)
max_i = np.max(np.argmax(dists, axis=0))
max_j = np.argmax(np.argmax(dists, axis=0))
print("Highest:", max_i, max_j, dists[max_i, max_j])

for cluster_size in [230]:
    seed_a = data_a[max_i]
    seed_b = data_b[max_j]
    print("Seeds:", euclidean_distances([seed_a], [seed_b]))

    dists_a = euclidean_distances([seed_a], data_a)[0]
    cluster_a = dists_a.argsort()[1:cluster_size+1]
    # print(dists_a[cluster_a])
    print(data_a[cluster_a].shape)
    
    dists_b = euclidean_distances([seed_b], data_b)[0]
    cluster_b = dists_b.argsort()[1:cluster_size+1]
    # print(dists_b[cluster_b])
    print(data_b[cluster_b].shape)

    model = TSNE(n_components=2, init="pca", learning_rate="auto")
    data_tsne = model.fit_transform(np.concatenate((data_a[cluster_a], data_b[cluster_b])))
    assert len(data_tsne) == 2*cluster_size
    data_a_tsne = data_tsne[:cluster_size]
    data_b_tsne = data_tsne[cluster_size:]

    plt.scatter(
        [x[0] for x in data_a_tsne],
        [x[1] for x in data_a_tsne],
    )
    plt.scatter(
        [x[0] for x in data_b_tsne],
        [x[1] for x in data_b_tsne],
    )
    plt.show()