#!/usr/bin/env python3
import os
import random
import json
import pickle
import pandas as pd

def extract_from_coco(loc,to_select,data_path):
    data = json.load(open(loc))
    annotations = data["annotations"]
    captions = []
    for caption in annotations:
        cap = caption["caption"]
        if len(cap.split())!=0:
            captions.append(cap.strip())
    namb = random.sample(captions, to_select)
    namb_data_path = os.path.join(os.getcwd(),os.path.join("data","ambiguity"))
    h = open(os.path.join(data_path,"unambiguous_coco.txt"),"w")
    for sent in namb:
        print(sent,file=h)
    return namb

def sentence_ambiguous(data_loc):
    data_path = os.path.join(data_loc,"ambiguity")
    amb_sent = os.path.join(data_path,"ambiguous_coco.txt")
    ambiguous = open(amb_sent,"r").read().split("\n")
    amb = []
    for sent in ambiguous:
       if len(sent.split())!=0:
           amb.append(sent)
    
    namb_sent = os.path.join(data_path,"unambiguous_coco.txt")
    
    if not os.path.exists(namb_sent):
        len_ambig = len(amb)
        namb_path = os.path.join(data_path,os.path.join("annotations_trainval2017",os.path.join("annotations","captions_train2017.json")))
        namb = extract_from_coco(namb_path,len_ambig,data_path)
    
    unambiguous = open(namb_sent,"r").read().split("\n")
    namb = []
    for sent in unambiguous:
       if len(sent.split())!=0:
           namb.append(sent)
    
    return amb,namb

def pickle_saver(folder_loc,model_name,rep_list):
    rep_name = model_name.upper()+".pkl"
    file_loc = os.path.join(folder_loc,rep_name)
    with open(file_loc, 'wb') as f:
        pickle.dump(rep_list, f)
    

def emmt(loc):
    data = open(loc,"r").read().split("\n")
    Amb, Namb = [],[]
    for sent in data:
        sp = sent.split(",")
        if len(sp)==3:
            cat = sp[0]
            text = sp[1]
            if cat=="namb":
                Namb.append(text)
            else:
                Amb.append(text)
    return Amb, Namb

def read_blimp_cases(folder_loc):
    s_good = []
    g_file = open(os.path.join(folder_loc,"s_good")).read().split("\n")
    b_file = open(os.path.join(folder_loc,"s_bad")).read().split("\n")
    return g_file[:-1],b_file[:-1]

def CoLA_extract(data_folder):
    Accept = []
    Unaccept = []
    
    CoLA_loc = os.path.join(data_folder,os.path.join("grammaticality",
    os.path.join("cola_public_1.1",os.path.join("cola_public","raw"))))
    file_name = "in_domain_train.tsv"
    content = pd.read_csv(os.path.join(CoLA_loc,file_name), sep='\t',names=['Source', 'acceptability judgment', 'author annotation', 'sentence'])
    acceptable = content.loc[content['acceptability judgment'] == 1]
    unacceptable = content.loc[content['acceptability judgment'] == 0]
    accept = acceptable["sentence"].to_list()
    unaccept = unacceptable["sentence"].to_list()
    Accept = Accept+accept
    Unaccept = Unaccept+unaccept
    
    file_name = "in_domain_dev.tsv"
    content = pd.read_csv(os.path.join(CoLA_loc,file_name), sep='\t',names=['Source', 'acceptability judgment', 'author annotation', 'sentence'])
    acceptable = content.loc[content['acceptability judgment'] == 1]
    unacceptable = content.loc[content['acceptability judgment'] == 0]
    accept = acceptable["sentence"].to_list()
    unaccept = unacceptable["sentence"].to_list()
    Accept = Accept+accept
    Unaccept = Unaccept+unaccept

    Accept = random.sample(Accept, len(Unaccept))
    
    return Accept,Unaccept

def Complexity_extract(data_folder,language):

    data_loc = os.path.join(data_folder,os.path.join("complexity"))
    file_name = "simple_"+language
    content = open(os.path.join(data_loc,file_name),"r").read().split("\n")
    simple = content[:-1]

    data_loc = os.path.join(data_folder,os.path.join("complexity"))
    file_name = "complicated_"+language
    content = open(os.path.join(data_loc,file_name),"r").read().split("\n")
    complicated = content[:-1]
    
    return simple,complicated