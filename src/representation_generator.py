#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from representation_generator_utils import sentence_ambiguous, pickle_saver, emmt, read_blimp_cases, CoLA_extract
from transformers import AutoTokenizer, AutoModel
import numpy as np

class Model_sent:
    def __init__(self, model):
        if model=='bert':
            self.model = "bert-base-multilingual-cased"
        elif model=='gpt2':
            self.model = 'gpt2'
        elif model=='sbert':
            self.model="DeepPavlov/bert-base-multilingual-cased-sentence"
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


    def get_representations(self,model_name,sent,class_tag):
        
        tok = self.tok(sent) #shape=[1,w]
        
        if model_name=='sbert' or model_name=='bert':
            """
            Dim-0: [1,n_words,768]
            Dim-2: [1,768] a.k.a pooler (mean)
            Dim-3: 13*[1,n_words,768]
            """
            final_layer_tokens, pooler, hidden_states = self.word_repr(tok,pooler=True) 
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
        

        if model_name=='bert' or model_name=='sbert': 
            representation = {"sent":sent,"mean":Mean,"haddamard":Haddamard,"sum":Sum,"class":class_tag,"pooler":pooler_rep,"cls":CLS_List}
    
        if model_name=='gpt2': 
            representation = {"sent":sent,"mean":Mean,"haddamard":Haddamard,"sum":Sum,"class":class_tag}
        
        return representation
        
        

def create_representations(model_name,sentence_list,model,class_tag):    
    Representations = []
    for sent in sentence_list:
        representation = model.get_representations(model_name,sent,class_tag)
        Representations.append(representation)

    return Representations  


def model_block(model_name,sent_set1,sent_set2,class_tag1,class_tag2,folder_loc):

    model = Model_sent(model_name)
    tok, o = model.init_model()
    m_name = model_name
    Representation_Set = []

    #Format: model_name,dataset,model_file,storage_location,class
    Rep = create_representations(model_name,sent_set1,model,class_tag1)
    Representation_Set = Representation_Set + Rep
    Rep = create_representations(model_name,sent_set2,model,class_tag2)
    Representation_Set = Representation_Set + Rep
    pickle_saver(folder_loc,model_name,Representation_Set)

def ambiguity_representation_generator(dataset_name,amb,namb):
    rep_folder = os.path.join(os.getcwd(),"Representations")
    ambiguity_loc = os.path.join(rep_folder,"Ambiguity")
    if not os.path.exists(ambiguity_loc):
        print("Creating new folder to store the representations")
        os.mkdir(ambiguity_loc)
    folder_loc = os.path.join(ambiguity_loc,dataset_name)
    if not os.path.exists(folder_loc):
        print("Creating new folder to store the representations")
        os.mkdir(folder_loc)
    
    model_block("bert",amb,namb,"A","U",folder_loc)
    model_block("gpt2",amb,namb,"A","U",folder_loc)
    model_block("sbert",amb,namb,"A","U",folder_loc)

def blimp_representation_generator(root_folder,dataset_name,s_good,s_bad):
    rep_folder = os.path.join(os.getcwd(),"Representations")
    if not os.path.exists(os.path.join(rep_folder,"Grammaticality")):
        print("Creating new folder to store the representations")
        os.mkdir(os.path.join(rep_folder,"Grammaticality"))
    folder_loc = os.path.join(os.path.join(rep_folder,"Grammaticality"),root_folder)
    if not os.path.exists(folder_loc):
        print("Creating new folder to store the representations")
        os.mkdir(folder_loc)
    folder_loc = os.path.join(os.path.join(rep_folder,"Grammaticality"),os.path.join(root_folder,dataset_name))
    if not os.path.exists(folder_loc):
        print("Creating new folder to store the representations")
        os.mkdir(folder_loc)

    model_block("bert",s_good,s_bad,"G","B",folder_loc)
    model_block("gpt2",s_good,s_bad,"G","B",folder_loc)
    model_block("sbert",s_good,s_bad,"G","B",folder_loc)
   
def CoLA_representation_generator(dataset_name,s_good,s_bad):
    rep_folder = os.path.join(os.getcwd(),"Representations")
    grammaticality_loc = os.path.join(rep_folder,"Grammaticality")
    folder_loc = os.path.join(grammaticality_loc,dataset_name)
    if not os.path.exists(folder_loc):
        print("Creating new folder to store the representations")
        os.mkdir(folder_loc)
    
    model_block("bert",s_good,s_bad,"G","B",folder_loc)
    model_block("gpt2",s_good,s_bad,"G","B",folder_loc)
    model_block("sbert",s_good,s_bad,"G","B",folder_loc)

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("-d", "--data")
    args = args.parse_args()

    data_location = args.data

    rep_folder =  os.path.join(os.getcwd(),"Representations")
    if not os.path.exists(rep_folder):
        print("Creating new folder to store the representations")
        os.mkdir(rep_folder)

    """Ambiguity Representations"""
    print("Processing ambiguity category : COCO")
    amb,namb = sentence_ambiguous(args.data)   
    ambiguity_representation_generator("COCO",amb,namb)
    
    print("Processing ambiguity category : EMMT")
    amb, namb = emmt(os.path.join("data",os.path.join("ambiguity",os.path.join("EMMT","sentence_list.csv"))))
    ambiguity_representation_generator("EMMT",amb,namb)
    
    """Grammaticality Representations"""
    
    grammaticality_data_loc = os.path.join(data_location,"grammaticality")
    categories = ["morphology","syntax","semantics","syntax_semantics"]
    for category in categories:
        data_dir = os.path.join(grammaticality_data_loc,category) 
        for files in os.listdir(data_dir):
            print("Processing grammaticality category :%s "%files)
            s_good,s_bad = read_blimp_cases(os.path.join(data_dir,files))
            blimp_representation_generator(category,files,s_good,s_bad)         

    acceptable, unacceptable = CoLA_extract(data_location)
    CoLA_representation_generator("CoLA",acceptable,unacceptable)