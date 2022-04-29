#!/usr/bin/env python3

from glob import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle

for model in ["BERT", "GPT"]:
    data = []
    for amb_mode in ["ambiguous", "unambiguous"]:
        with open(f"data/Ambiguity_EMMT/{model}/{amb_mode}_representations_EMMT/sentence_representations.pkl", "rb") as f:
            data_local = pickle.load(f)
            print(len(data_local))

            for sent in data_local:
                data.append({
                    "sent": sent["sent"],
                    "mean": sent["Token_Aggregation"]["Mean"],
                    "haddamard": sent["Token_Aggregation"]["Haddamard"],
                    "sum": sent["Token_Aggregation"]["Sum"],
                    "amb": amb_mode == "ambiguous"
                })
                if model != "GPT":
                    data[-1]["pooler"] = sent["pooler"]
                    data[-1]["cls"] = sent["CLS"]

    print("Loaded", len(data), "data")
    print("Every entry has the following keys:", data[0].keys())

    with open(f"data/ambiguity_EMMT_{model}.pkl", "wb") as f:
        pickle.dump(data, f)
