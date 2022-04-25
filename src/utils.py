import pickle
import json

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
