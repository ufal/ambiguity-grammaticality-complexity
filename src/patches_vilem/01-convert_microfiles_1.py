#!/usr/bin/env python3

from glob import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle

data = defaultdict(lambda: {"cls": [], "mean": [], "haddamard": [], "sum": []})

MODEL = "GPT"

for layer in tqdm(range(12)):
    for amb_mode in ["ambiguous", "unambiguous"]:
        for f in glob(f"data/Ambiguity_COCO/{MODEL}/{amb_mode}_representations/CLS/{layer}/*.npy"):
            fname = f.split("/")[-1].split(".npy")[0]
            with open(f, "rb") as f:
                v = np.load(f)
                data[fname]["cls"].append(v)

                # 1st attempt
                data[fname]["amb"] = amb_mode == "ambiguous"


        for f in glob(f"data/Ambiguity_COCO/{MODEL}/{amb_mode}_representations/Token_Aggregation/Haddamard/{layer}/*.npy"):
            fname = f.split("/")[-1].split(".npy")[0]
            with open(f, "rb") as f:
                v = np.load(f)
                data[fname]["haddamard"].append(v)

        for f in glob(f"data/Ambiguity_COCO/{MODEL}/{amb_mode}_representations/Token_Aggregation/Mean/{layer}/*.npy"):
            fname = f.split("/")[-1].split(".npy")[0]
            with open(f, "rb") as f:
                v = np.load(f)
                data[fname]["mean"].append(v)

        for f in glob(f"data/Ambiguity_COCO/{MODEL}/{amb_mode}_representations/Token_Aggregation/Sum/{layer}/*.npy"):
            fname = f.split("/")[-1].split(".npy")[0]
            with open(f, "rb") as f:
                v = np.load(f)
                data[fname]["sum"].append(v)

                # 2nd attempt
                data[fname]["amb"] = amb_mode == "ambiguous"

        # a hack to be included in tqdm
        if layer == 0:
            for f in glob(f"data/Ambiguity_COCO/{MODEL}/ambiguous_representations/pooler/*.npy"):
                fname = f.split("/")[-1].split(".npy")[0]
                with open(f, "rb") as f:
                    v = np.load(f)
                    data[fname]["pooler"] = v

data = [
    {"name": fname, **vals}
    for fname, vals in data.items()
]

print("Loaded", len(data), "data")
print("Every entry has the following keys:", data[0].keys())

with open(f"data/ambiguity_coco_{MODEL}.pkl", "wb") as f:
    pickle.dump(data, f)