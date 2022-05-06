import pickle
import json
import os


def read_json(path):
    with open(path, "r") as fread:
        return json.load(fread)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def read_pickle(path):
    with open(path, "rb") as fread:
        reader = pickle.Unpickler(fread)
        return reader.load()


def save_pickle(path, data):
    with open(path, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        pickler.dump(data)


def read_tfidf(name):
    with open('computed/tfidf_baselines.tsv', "r") as f:
        for line in f.readlines():
            items = line.strip().split("\t")
            if len(items) < 2:
                continue
            if items[0] == name:
                return items[1]

def read_tfidf_neural(name):
    with open('computed/tfidf_baselines_mlp.tsv', "r") as f:
        for line in f.readlines():
            items = line.strip().split("\t")
            if len(items) < 2:
                continue
            if items[0] == name:
                return items[1]

def file_list(name):
    return [os.path.join(name, "BERT.pkl"), os.path.join(name, "GPT2.pkl"), os.path.join(name, "SBERT.pkl")]


def create_folders(path):
    if not os.path.exists(path):
        os.mkdir(path)


def computed_path_generator(abs_path):
    items = abs_path.split("/")
    rep_index = items.index('Representations')
    path_list = items[rep_index + 1:-1]
    target_path = os.path.join(os.getcwd(), "computed")
    create_folders(target_path)
    for locations in path_list:
        target_path = os.path.join(target_path, locations)
        create_folders(target_path)
    return target_path


def folder_generator_graph(abs_path):
    items = abs_path.split("/")
    rep_index = items.index('graphs')
    path_list = items[rep_index + 1:]
    # print(path_list)
    target_path = os.path.join(os.getcwd(), "graphs")
    create_folders(target_path)
    for locations in path_list:
        target_path = os.path.join(target_path, locations)
        create_folders(target_path)
