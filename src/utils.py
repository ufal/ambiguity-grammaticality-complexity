import pickle
import json
import os

def load_amb(model="BERT", dataset="EMMT"):
    model = model.upper()
    dataset = dataset.upper()
    with open(f"data/ambiguity_{dataset}_{model}.pkl", "rb") as f:
        return pickle.load(f)

def load_gr(model="BERT", task="irregular_plural_subject_verb_agreement_2"):
    with open(f"data/gr_{task}_{model}.pkl", "rb") as f:
        return pickle.load(f)

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
    f_ = open(os.path.join(os.getcwd(),"computed/tfidf_baselines.tsv"),"r").read().split("\n")
    for item in f_:
        items = item.split("\t")
        if len(items)==2:
            name_item = items[0]
            value = items[1]
            if name_item==name:
                break
    return value

def file_list(name):
    return [os.path.join(name,"BERT.pkl"),os.path.join(name,"GPT2.pkl"),os.path.join(name,"SBERT.pkl")]

def create_folders(path):
    if not os.path.exists(path):
        os.mkdir(path)

def computed_path_generator(abs_path):
    items = abs_path.split("/")
    rep_index = items.index('Representations')
    path_list = items[rep_index+1:-1]
    target_path = os.path.join(os.getcwd(),"computed")
    create_folders(target_path)
    for locations in path_list:
        target_path = os.path.join(target_path,locations)
        create_folders(target_path)
    return target_path

def folder_generator_graph(abs_path):
    items = abs_path.split("/")
    rep_index = items.index('graphs')
    path_list = items[rep_index+1:]
    # print(path_list)
    target_path = os.path.join(os.getcwd(),"graphs")
    create_folders(target_path)
    for locations in path_list:
        target_path = os.path.join(target_path,locations)
        create_folders(target_path)
    