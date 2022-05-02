#!/usr/bin/env python3

from utils import read_pickle
from argparse import ArgumentParser
args = ArgumentParser()
args.add_argument("-d", "--data", default="data/CoLA_BERT.pkl")
args = args.parse_args()

data = read_pickle(args.data)
sample = data[13]
print(sample.keys())