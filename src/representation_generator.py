#!/usr/bin/env python3
import os
import random
import json
import shutil
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd

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

def extract_from_coco(loc,to_select):
    data = json.load(open(loc))
    annotations = data["annotations"]
    captions = []
    for caption in annotations:
        cap = caption["caption"]
        if len(cap.split())!=0:
            captions.append(cap.strip())
    random.seed(1314)
    namb = random.sample(captions, to_select)
    namb_data_path = os.path.join(os.getcwd(),os.path.join("data","ambiguity"))
    h = open(os.path.join(namb_data_path,"unambiguous_coco.txt"),"w")
    for sent in namb:
        print(sent,file=h)
    return namb

def sentence_ambiguous():
    data_path = os.path.join(os.getcwd(),os.path.join("data","ambiguity"))
    amb_sent = os.path.join(data_path,"ambiguous_coco.txt")
    ambiguous = open(amb_sent,"r").read().split("\n")
    amb = []
    for sent in ambiguous:
       if len(sent.split())!=0:
           amb.append(sent)
    len_ambig = len(amb)

    namb_path = os.path.join(data_path,os.path.join("annotations_trainval2017",os.path.join("annotations","captions_train2017.json")))
    namb = extract_from_coco(namb_path,len_ambig)

    return amb,namb

class Model_sent:
    def __init__(self, model):
        if model=='bert':
            self.model = "bert-base-cased"
        elif model=='gpt2':
            self.model = 'gpt2'
        elif model=='sbert':
            self.model="sentence-transformers/bert-base-nli-mean-tokens"
        self.max_len = 100

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model,cache_dir="huggingface_cache")  
        self.model_out = AutoModel.from_pretrained(self.model, output_hidden_states=True,cache_dir="huggingface_cache")
        return self.tokenizer,self.model_out
    
    def tok(self,sent):
        encoded = self.tokenizer.encode_plus(text=sent,add_special_tokens=True,return_tensors = 'pt')
        return encoded['input_ids']
    
    def word_repr(self,word,pooler=False):
        pooler_output = None
        
        outputs = self.model_out(word)

        if not pooler:
            hidden_states = outputs.hidden_states
            last_hidden = outputs.last_hidden_state

        if pooler: #FOR BERT OUTPUTS
            pooler_output = outputs[1] #shape=[1,768] --> hidden state corresponding to the first token
            last_hidden = outputs[0] # shape=[1,w,768]...final output       
            hidden_states = outputs[2] #len=13....0th layer output is representation from embedding layer
        
        return last_hidden,pooler_output,hidden_states
    
    def hadd_repr(self,repr):
        p = repr[0]
        for it in range(1,repr.shape[0]):
            r1 = repr[it]
            p = np.multiply(p,r1,dtype=np.float64)
        return p


    def get_grammaticality_representations(self,model_name,sent,s_tag):
        tok = self.tok(sent) #shape=[1,w]
        
        if model_name=='bert':
            _, pooler, hidden_states = self.word_repr(tok,pooler=True)
            pooler_representation = pooler[0].cpu().detach().numpy()
            pooler_rep = np.array(pooler_representation,dtype=np.float64)
            CLS=True
            CLS_List = []

        elif model_name=='gpt2':
            _, pooler, hidden_states = self.word_repr(tok)
            CLS = False #GPT outputs will not have Pooler or CLS
            

        Mean, Sum, Haddamard = [],[],[]
        for lyr in range(12):
            
            layer_rep = hidden_states[lyr][0]
                
            if CLS:
                CLS_representation = layer_rep[0].cpu().detach().numpy()
                CLS_rep = np.array(CLS_representation,dtype=np.float64)
                CLS_List.append(CLS_rep)
 
            mean_representation = layer_rep.mean(axis=0).cpu().detach().numpy()
            mean_rep = np.array(mean_representation,dtype=np.float64)
            Mean.append(mean_rep)
 
            sum_representation = np.sum(layer_rep.cpu().detach().numpy(),axis=0)
            sum_rep = np.array(sum_representation,dtype=np.float64)
            Sum.append(sum_rep)

            hadd_representation = self.hadd_repr(layer_rep.cpu().detach().numpy()) 
            hadd_rep = np.array(hadd_representation,dtype=np.float64)
            Haddamard.append(hadd_rep)
        

        if model_name=='bert': 
            representation = {"sent":sent,"mean":Mean,"haddamard":Haddamard,"sum":Sum,"class":s_tag,"pooler":pooler_rep,"cls":CLS_List}
    
        if model_name=='gpt2': 
            representation = {"sent":sent,"mean":Mean,"haddamard":Haddamard,"sum":Sum,"class":s_tag}
        
        return representation

    def get_representations(self,model_name,model_rep_dir,sent,name,amb_tag):
        pooler_dir_name = os.path.join(model_rep_dir,'Pooler')
        cls_dir_name = os.path.join(model_rep_dir,'CLS')
        tok_agg_dir_name = os.path.join(model_rep_dir,'Token_Aggregation')
        tok = self.tok(sent) #shape=[1,w]
        
        if model_name=='bert':
            _, pooler, hidden_states = self.word_repr(tok,pooler=True)
            pooler_representation = pooler[0].cpu().detach().numpy()
            pooler_loc = os.path.join(pooler_dir_name,name)
            pooler_rep = np.array(pooler_representation,dtype=np.float64)
            CLS=True
            CLS_List = []

        elif model_name=='gpt2':
            _, pooler, hidden_states = self.word_repr(tok)
            CLS = False #GPT outputs will not have Pooler or CLS
            

        Mean, Sum, Haddamard = [],[],[]
        for lyr in range(12):
            cls_loc = os.path.join(cls_dir_name,str(lyr))
            mean_loc = os.path.join(tok_agg_dir_name,os.path.join("Mean",str(lyr)))
            sum_loc = os.path.join(tok_agg_dir_name,os.path.join("Sum",str(lyr)))
            haddamard_loc = os.path.join(tok_agg_dir_name,os.path.join("Haddamard",str(lyr)))
            
            layer_rep = hidden_states[lyr][0]
                
            if CLS:
                CLS_representation = layer_rep[0].cpu().detach().numpy()
                CLS_loc = os.path.join(os.path.join(cls_dir_name,str(lyr)),name)
                CLS_rep = np.array(CLS_representation,dtype=np.float64)
                CLS_List.append(CLS_rep)
 
            mean_representation = layer_rep.mean(axis=0).cpu().detach().numpy()
            mean_rep = np.array(mean_representation,dtype=np.float64)
            Mean.append(mean_rep)
 
            sum_representation = np.sum(layer_rep.cpu().detach().numpy(),axis=0)
            sum_rep = np.array(sum_representation,dtype=np.float64)
            Sum.append(sum_rep)

            hadd_representation = self.hadd_repr(layer_rep.cpu().detach().numpy()) 
            hadd_rep = np.array(hadd_representation,dtype=np.float64)
            Haddamard.append(hadd_rep)
        

        if model_name=='bert': 
            representation = {"sent":sent,"mean":Mean,"haddamard":Haddamard,"sum":Sum,"amb":amb_tag,"pooler":pooler_rep,"cls":CLS_List}
    
        if model_name=='gpt2': 
            representation = {"sent":sent,"mean":Mean,"haddamard":Haddamard,"sum":Sum,"amb":amb_tag}
        
        return representation
        

def count_format(count):
    if count<100 :
        count_str = "0"+str(count)
        if count<10:
            count_str = "0"+str(count_str)
        return count_str
    else:
        return str(count)
  

def create_folder_structure(test_type,model_type):
    dir_name = os.path.join(os.getcwd(),os.path.join("Representations",os.path.join(test_type,model_type)))
    rep_dir = dir_name
    shutil.rmtree(rep_dir, ignore_errors=True)
    os.makedirs(rep_dir)
    return rep_dir

def create_grammaticality_folders(loc):
    shutil.rmtree(loc, ignore_errors=True)
    os.makedirs(loc)
    return loc
    

def create_representations(f_name,sentence_list,model,m_name,rep_dir,amb_tag):    
    count_s = 0
    f_ = open(os.path.join(os.getcwd(),f_name),"w")
    f_.write("id\tsentence\n")
    Representations = []
    for sent in sentence_list:
        sid = "a"+count_format(count_s+1)
        if f_name.split("_")[0]=="unambiguous":
            sid = "u"+count_format(count_s+1)
        h = sid+"\t"+sent
        print(h)
        print(h,file=f_)
        if m_name=='sbert':
            embeddings = model.encode(sent)
            representation = {"sent":sent,"embedding":embeddings,"amb":amb_tag}
            Representations.append(representation)
        else:    
            representation = model.get_representations(m_name,rep_dir,sent,sid,amb_tag)
            Representations.append(representation)
        count_s+=1

    return Representations  


def ambiguity_representation_generator(dataset_name,amb,namb):
    model_bert = Model_sent('bert')
    bert_tok, bert_o = model_bert.init_model()
    m_name = 'bert'

    BERT = []
    
    rep_dir = create_folder_structure("Ambiguity","BERT_"+dataset_name)
    Rep = create_representations("ambiguous_sid_bert_"+dataset_name,amb,model_bert,m_name,rep_dir,"A")
    
    BERT = BERT + Rep

    Rep = create_representations("unambiguous_sid_bert_"+dataset_name,namb,model_bert,m_name,rep_dir,"U")
    
    BERT = BERT + Rep

    f_name  = os.path.join(rep_dir,'sentence_representations.pkl')
    with open(f_name, 'wb') as f:
        pickle.dump(BERT, f)


    model_gpt = Model_sent('gpt2')
    gpt_tok, gpt_o = model_gpt.init_model()
    m_name='gpt2'

    GPT = []

    rep_dir = create_folder_structure("Ambiguity","GPT_"+dataset_name)
    Rep = create_representations("ambiguous_sid_gpt_"+dataset_name,amb,model_gpt,m_name,rep_dir,"A")
    
    GPT = GPT + Rep

    Rep = create_representations("unambiguous_sid_gpt_"+dataset_name,namb,model_gpt,m_name,rep_dir,"U")
    
    GPT = GPT + Rep


    f_name  = os.path.join(rep_dir,'sentence_representations.pkl')
    with open(f_name, 'wb') as f:
        pickle.dump(GPT, f)

    model_sbert =SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    m_name='sbert'

    SBERT = []

    rep_dir = create_folder_structure("Ambiguity","SBERT_"+dataset_name)
    Rep = create_representations("ambiguous_sid_sbert_"+dataset_name,amb,model_sbert,m_name,rep_dir,"A")
    
    SBERT = SBERT + Rep

    Rep = create_representations("unambiguous_sid_sbert_"+dataset_name,namb,model_sbert,m_name,rep_dir,"U")

    SBERT = SBERT + Rep


    f_name  = os.path.join(rep_dir,'sentence_representations.pkl')
    with open(f_name, 'wb') as f:
        pickle.dump(SBERT, f)


def create_grammaticality_representations(sentence_list,model,m_name,s_tag):    
    Representations = []
    for sent in sentence_list:
        if m_name=='sbert':
            embeddings = model.encode(sent)
            representation = {"sent":sent,"embedding":embeddings,"class":s_tag}
            Representations.append(representation)
        else:    
            representation = model.get_grammaticality_representations(m_name,sent,s_tag)
            Representations.append(representation)

    return Representations  

def grammaticality_representation_generator(dataset_loc,s_good,s_bad):
    print("bert representations")
    model_bert = Model_sent('bert')
    bert_tok, bert_o = model_bert.init_model()
    m_name = 'bert'

    BERT = []
    
    Rep = create_grammaticality_representations(s_good,model_bert,m_name,"G")
    
    BERT = BERT + Rep

    Rep = create_grammaticality_representations(s_bad,model_bert,m_name,"B")

    
    BERT = BERT + Rep

    f_name  = os.path.join(dataset_loc,'sentence_representations.pkl')
    with open(f_name, 'wb') as f:
        pickle.dump(BERT, f)

    print("gpt representations")
    model_gpt = Model_sent('gpt2')
    gpt_tok, gpt_o = model_gpt.init_model()
    m_name='gpt2'

    GPT = []

    Rep = create_grammaticality_representations(s_good,model_bert,m_name,"G")
    
    GPT = GPT + Rep

    Rep = create_grammaticality_representations(s_bad,model_bert,m_name,"B")
    
    GPT = GPT + Rep


    f_name  = os.path.join(dataset_loc,'sentence_representations.pkl')
    with open(f_name, 'wb') as f:
        pickle.dump(GPT, f)

    print("sbert representations")
    model_sbert =SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    m_name='sbert'

    SBERT = []

    Rep = create_grammaticality_representations(s_good,model_sbert,m_name,"G")
    
    SBERT = SBERT + Rep

    Rep = create_grammaticality_representations(s_bad,model_sbert,m_name,"B")

    SBERT = SBERT + Rep


    f_name  = os.path.join(dataset_loc,'sentence_representations.pkl')
    with open(f_name, 'wb') as f:
        pickle.dump(SBERT, f)




def CoLA_extract():
    CoLA_loc = os.path.join(os.getcwd(),os.path.join("data",os.path.join("grammaticality",
    os.path.join("cola_public_1.1",os.path.join("cola_public",os.path.join("raw"))))))
    file_name = "in_domain_dev.tsv"
    content = pd.read_csv(os.path.join(CoLA_loc,file_name), sep='\t',names=['Source', 'acceptability judgment', 'author annotation', 'sentence'])
    acceptable = content.loc[content['acceptability judgment'] == 1]
    unacceptable = content.loc[content['acceptability judgment'] == 0]
    return acceptable,unacceptable


if __name__ == "__main__":
    
    amb, namb = sentence_ambiguous()
    ambiguity_representation_generator("COCO",amb,namb)
    amb, namb = emmt(os.path.join("data",os.path.join("ambiguity",os.path.join("EMMT","sentence_list.csv"))))
    ambiguity_representation_generator("EMMT",amb,namb)

    grammatical_loc = os.path.join(os.getcwd(),os.path.join("data","grammaticality"))
    rep_loc = os.path.join(os.getcwd(),os.path.join("Representations","Grammaticality"))
    create_grammaticality_folders(rep_loc)

    categories = ["morphology","syntax","semantics","syntax_semantics"]
    for category in categories:
        data_dir = os.path.join(grammatical_loc,category) 
        rep_dir = os.path.join(rep_loc,category)
        create_grammaticality_folders(rep_dir)
        for files in os.listdir(data_dir):
            print("category :%s extracting"%files)
            target_data_folder = os.path.join(data_dir,files)
            rep_data_folder = os.path.join(rep_dir,files)
            create_grammaticality_folders(rep_data_folder)
            s_good = os.path.join(target_data_folder,"s_good")
            s_bad = os.path.join(target_data_folder,"s_bad")
            grammaticality_representation_generator(rep_data_folder,s_good,s_bad)
            print("category :%s done"%files)

    cola_loc = os.path.join(rep_loc,"CoLA")
    create_grammaticality_folders(cola_loc)
    acceptable, unacceptable = CoLA_extract()
    grammaticality_representation_generator(cola_loc,acceptable,unacceptable)











    
