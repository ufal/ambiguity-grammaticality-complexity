#!/usr/bin/env python3

from utils import load_amb, load_gr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import shuffle

# data = load_amb(model="BERT")
data = load_gr(model="bert", task="morphology_irregular_plural_subject_verb_agreement_2")


for layer in range(12):
    # penalty="elasticnet", solver="saga", l1_ratio=0.5
    model = LogisticRegression(C=0.3)

    data_x = [x["mean"][layer] for x in data]
    data_x = StandardScaler().fit_transform(data_x)
    # data_y = [x["amb"] for x in data]
    data_y = [x["class"] for x in data]

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, test_size=20, shuffle=True, random_state=0,
    )

    model.fit(data_x_train, data_y_train)
    score_train = model.score(data_x_train, data_y_train)
    score_test = model.score(data_x_test, data_y_test)

    print(f"Layer {layer} accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}")
