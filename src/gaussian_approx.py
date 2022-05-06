"""
Gaussian approximation of layer representations to evaluate goodness of fit
"""
import sklearn.mixture
import os
from utils import read_json, read_pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def gmm_func(vec):
    dimensions = vec.shape[1]
    Std = []
    for dimension in range(dimensions):
        data_sample = vec[:,dimension]
        mean,std=norm.fit(data_sample)
        Std.append(std)
    print("Average standard deviation across dimensions %f"%np.mean(np.array(Std)))
    

def generate_estimation(rep_loc,name_model,s_tag1,s_tag2):
    data = read_pickle(rep_loc)
    
    if name_model=="BERT" or name_model=="SBERT":
        rep_list = ["mean", "haddamard", "sum", "pooler", "cls"]
    else:
        rep_list = ["mean", "haddamard", "sum"]
    ct = 0
    Pooler1 = []
    Pooler2 = []
    for index in range(len(data)):
        x = data[index]
        pooler = x["pooler"] 
        if x["class"]==s_tag1:
            Pooler1.append(pooler)
        else:
            Pooler2.append(pooler)
    gmm_func(np.array(Pooler1))
    gmm_func(np.array(Pooler2))
            


def Gaussian_Estimator(loc,s_tag1,s_tag2):
    BERT = os.path.join(loc,"BERT.pkl")
    GPT = os.path.join(loc,"GPT2.pkl")
    SBERT = os.path.join(loc,"SBERT.pkl")
    
    generate_estimation(BERT,"BERT","A","U")



if __name__=="__main__":
    repr_home = os.path.join(os.getcwd(),"Representations")
    
    """Ambiguity"""
    amb_loc = os.path.join(repr_home,"Ambiguity")

    COCO_loc = os.path.join(amb_loc,"COCO")
    Gaussian_Estimator(COCO_loc,"A","U")