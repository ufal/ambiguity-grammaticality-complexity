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

def read_tfidf(name,target):
    if target=="amb":
        name = name.split("_")[2].replace(".json","")
        target_name=name
    else:
        file_name=''
        name_components = name.split("_")
        for c in range(1,len(name_components)-1):
            if c!=len(name_components)-1:
                file_name+=name_components[c]+"_"
            else:
                file_name+=name_components[c]
        target_name = file_name 
    f_ = open(os.path.join(os.getcwd(),"tfidf_baselines"),"r").read().split("\n")
    for item in f_:
        items = item.split("\t")
        if len(items)==2:
            name_item = items[0]
            value = items[1]
            if name_item==name:
                break
    return value

def json_name(name,target):
    file_name=''
    if target=="amb":
        file_name = name.split("/")[-2]
    else:
        file_name = name.split("/")[-2]
        model_name = ((name.split("/")[-1].split("_"))[0]).upper()
        file_name = model_name+"_"+file_name
    return file_name