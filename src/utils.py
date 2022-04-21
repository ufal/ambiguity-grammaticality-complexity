import pickle

def load_amb(model="BERT"):
    with open(f"data/ambiguity_{model}.pkl", "rb") as f:
        return pickle.load(f)