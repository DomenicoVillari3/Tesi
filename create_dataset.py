import os
import re
import tensorflow
import numpy as np
from tensorflow.keras.utils import to_categorical
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

#funzione per craere il dizionario delle labels da file
def get_labels(file):
    with open(file, "r") as f:
        #lego le azioni disponibili dal file
        labels = f.read().splitlines()
        f.close()
    
    labels_dict = dict()
    index=0
    for label in labels:
        labels_dict[label]=index
        index+=1
    #print(labels_dict)
    return labels_dict

#funzione per trovare i file associati ad una label basandosi sul nome 
def filtra_directory(root_dir,filtro):
    files=[]
    pattern = rf'\b{filtro}_\d+\.npy$'
    for file in os.listdir(root_dir):
        if re.search(pattern,file):
            files.append(os.path.join(root_dir,file))
    return files

def get_max_frame(file="frames.txt"):
    with open(file, "r") as f:
        frames = [int(line.strip()) for line in f]
        f.close()
    return max(frames)

def read_file_npy(labels,source_dir="points"):
    x=[]
    y=[]
    

    for label in labels.keys():
        files=filtra_directory(source_dir,label)
        
        #recupero i landmarks e le azioni associate
        for file in files:
            landmarks=np.load(file)
            x.append(landmarks)
            #print(np.shape(landmarks))
            y.append(labels[label])
    
    #padding per avere misure omogenee
    x=pad_sequences(x,maxlen=None, padding='post', dtype='float32',value=-99)
    
    print(np.shape(x))
    print(np.shape(y))

    return x,y



#------------------------------------------------------------
labels=get_labels(file="labels.txt")
x,y=read_file_npy(labels=labels,source_dir="points")

np.save("x.npy",x)
np.save("y.npy",y)