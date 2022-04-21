import pickle

def load_amb(model="BERT", dataset="EMMT"):
    model = model.upper()
    dataset = dataset.upper()
    with open(f"data/ambiguity_{dataset}_{model}.pkl", "rb") as f:
        return pickle.load(f)