#!/usr/bin/env python3

from utils import read_pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from random import shuffle
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("-d", "--data", default="Representations/Ambiguity/COCO/BERT.pkl")
args = args.parse_args()

print("processing %s"%args.data)

data = read_pickle(args.data)

max_acc = 0
for max_features in [32, 64, 128, 256, 512, 768, 1024, 1536]:
    model = LogisticRegression()
    vectorizer = TfidfVectorizer(max_features=max_features)
    data_x = vectorizer.fit_transform([line["sent"] for line in data])

    data_y = [x["class"] for x in data]

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, test_size=100, shuffle=True, random_state=0,
    )

    model.fit(data_x_train, data_y_train)
    score_train = model.score(data_x_train, data_y_train)
    score_test = model.score(data_x_test, data_y_test)

    print(f"Features {max_features} accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}")
    if score_test>max_acc:
        max_acc=score_test

f_name = open("computed/tfidf_baselines.tsv","a")
set_name = args.data
case_name = set_name.split("/")[-2]
print("%s\t%s"%(case_name,max_acc),file=f_name)
f_name.close()
