#!/usr/bin/env python3

from utils import load_amb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import shuffle

data = load_amb(model="BERT")

for layer in range(12):
    model = MLPClassifier(random_state=0, hidden_layer_sizes=(500,200),early_stopping=True)

    data_x = [x["cls"][layer] for x in data]
    data_x = StandardScaler().fit_transform(data_x)
    data_y = [x["amb"] for x in data]

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, test_size=100, shuffle=True, random_state=0,
    )

    model.fit(data_x_train, data_y_train)
    score_train = model.score(data_x_train, data_y_train)
    score_test = model.score(data_x_test, data_y_test)

    print(f"Layer {layer} accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}")
