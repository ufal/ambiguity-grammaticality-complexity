#!/usr/bin/env python3

from utils import read_pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from random import shuffle
from argparse import ArgumentParser

from sklearn.model_selection import KFold
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

args = ArgumentParser()
args.add_argument("-d", "--data", default="data/ambiguity_EMMT_BERT.pkl")
args = args.parse_args()

print("processing %s"%args.data)

data = read_pickle(args.data)

max_acc = 0
max_acc_train = None
max_vocab_coefs = None

for max_features in [32, 64, 128, 256, 512, 768, 1024, 1536]:
    model = LogisticRegression()
    vectorizer = TfidfVectorizer(stop_words = None,max_features=max_features)
    data_x = vectorizer.fit_transform([line["sent"] for line in data])

    rev_vocab = {v:k for k,v in vectorizer.vocabulary_.items()}

    data_y = [x["class"] for x in data]

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, test_size=100, shuffle=True, random_state=0,
    )

    # model.fit(data_x_train, data_y_train)
    # score_train = model.score(data_x_train, data_y_train)
    # score_test = model.score(data_x_test, data_y_test)
    cv = cross_validate(model, data_x, data_y, cv=10, n_jobs=-1, return_train_score=True)
    score_train = cv["train_score"]
    score_test = cv["test_score"]
    score_train = np.mean(np.array(score_train))
    score_test = np.mean(np.array(score_test))
   

    print(f"Features {max_features} accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}")

    if score_test > max_acc:
        max_acc = score_test
        max_acc_train = score_train
#         max_vocab_coefs = [(rev_vocab[k],v) for k,v in enumerate(model.coef_[0])]


# print("\nWord features:")
# max_vocab_coefs.sort(key=lambda x: x[1])
# max_vocab_coefs_min = max_vocab_coefs[:10]
# max_vocab_coefs_max = max_vocab_coefs[-10:][::-1]
# print("Positive.:", ', '.join([f"{x[0]}: {x[1]:.2f}" for x in max_vocab_coefs_max]))
# print("Negative.: ", ', '.join([f"{x[0]}: {x[1]:.2f}" for x in max_vocab_coefs_min]))

f_name = open("computed/tfidf_baselines.tsv","a")
f_name2 = open("computed/tfidf_baselines_train.tsv","a")
set_name = args.data
case_name = set_name.split("/")[-2]
print("%s\t%s"%(case_name,max_acc),file=f_name)
print("%s\t%s"%(case_name,max_acc_train),file=f_name2)
f_name.close()
f_name2.close()


print("MLP Classifier")
"""
MLP experiment
"""
max_acc = 0
max_acc_train = None
max_vocab_coefs = None

for max_features in [32, 64, 128, 256, 512, 768, 1024, 1536]:

    
    vectorizer = TfidfVectorizer(stop_words = None,max_features=max_features)
    data_x = vectorizer.fit_transform([line["sent"] for line in data])

   
    model = MLPClassifier(
        random_state=0,
        hidden_layer_sizes=(100,),
        early_stopping=True
    )

    # data_x = StandardScaler().fit_transform(data_x)
    data_y = [x["class"] for x in data]
    labels = np.unique(data_y)
    pre = preprocessing.LabelEncoder()
    pre.fit(labels)
    data_y = pre.transform(data_y)              

    cv = cross_validate(model, data_x, data_y, cv=10, n_jobs=-1, return_train_score=True)
    
    # data_y = [x["class"] for x in data]

    # data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    #     data_x, data_y, test_size=100, shuffle=True, random_state=0,
    # )

    # model.fit(data_x_train, data_y_train)
    # score_train = model.score(data_x_train, data_y_train)
    # score_test = model.score(data_x_test, data_y_test)
    score_train = cv["train_score"]
    score_test = cv["test_score"]
    score_train = np.mean(np.array(score_train))
    score_test = np.mean(np.array(score_test))

    print(f"Features {max_features} accuracy:\ttrain: {score_train:.2%}, test: {score_test:.2%}")

    if score_test > max_acc:
        max_acc = score_test
        max_acc_train = score_train

f_name = open("computed/tfidf_baselines_mlp.tsv","a")
f_name2 = open("computed/tfidf_baselines_train_mlp.tsv","a")
set_name = args.data
case_name = set_name.split("/")[-2]
print("%s\t%s"%(case_name,max_acc),file=f_name)
print("%s\t%s"%(case_name,max_acc_train),file=f_name2)
f_name.close()
f_name2.close()
