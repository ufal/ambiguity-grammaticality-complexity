#!/usr/bin/env python3

from utils import read_pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import shuffle
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("-d", "--data", default="data/ambiguity_COCO_BERT.pkl")
args.add_argument("--target", default="amb", help="Probably `amb` or `class`. Based on Sunit's export.")
args = args.parse_args()

data = read_pickle(args.data)

for layer in range(12):
    # penalty="elasticnet", solver="saga", l1_ratio=0.5
    model = LogisticRegression(C=0.3)

    data_x = [x["mean"][layer] for x in data]
    data_x = StandardScaler().fit_transform(data_x)
    data_y = [x[args.target] for x in data]

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, test_size=20, shuffle=True, random_state=0,
    )

    model.fit(data_x_train, data_y_train)
    score_train = model.score(data_x_train, data_y_train)
    score_test = model.score(data_x_test, data_y_test)

    print(f"Layer {layer} accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}")
