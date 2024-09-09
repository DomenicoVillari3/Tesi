'''
Va costruita una classe in grado di gestire gli individui
Le proprietà della classe sono: 
1) numero strati LSTM
2) dimensione degli strati LSTM
3)numero degli strati densi
4)dimensione degli strati densi
5) t addestramento
6) accuracy finale
'''
#DNA inteso come un dizionario del tipo:
#dna={"n_strati_LSTM":10,"dimensione_strati_LSTM":10,"n_strati_densi":10,"dimensione_strati_densi":10,"t_addestramento":10,"accuracy_finale":10,"fitness":0.1}




import random
from datetime import datetime
from NeuralNetwork import NeuralNetwork
import numpy as np
import math



class Individual:
    #costruttore
    def __init__(self,dna1=None,dna2=None):
        self.dna={}
        if dna1 != None and dna2 != None:
            #combina i 2 genitori
            self.combine_dna(dna1,dna2)
            #addestro il modello con i valori del dizionario
            
            
        elif dna1!= None:
            #copia il genitore
            self.dna=dna1
            
        else:
            #crea il nodo da 0 
            self.create_individual()
            #addestro il modello con i valori del dizionario
            
        #stampo dnaindividuidef.txt
        print(self.dna)
    

    def combine_dna(self,dna1,dna2):
        #array con i valori del dizionario dna
        indici=["n_strati_LSTM","dimensione_strati_LSTM","n_strati_densi","dimensione_strati_densi"]
        
        # Crea un nuovo dizionario per i valori combinati
        nuovo_dna = {}

        #variabili utili
        len_chiavi = len(indici)
        punto_taglio=random.randint(1,len_chiavi-1) #assicuro che almeno 1 gene provenga da entrambi i genitori

        #copio dal 1o genitore
        for i in range(punto_taglio):
            nuovo_dna[indici[i]]=dna1[indici[i]]
        #copio dal 2o genitore
        for i in range(punto_taglio,len_chiavi):
            nuovo_dna[indici[i]]=dna2[indici[i]]
        
        #imposto valori da calcolare
        self.dna=nuovo_dna
        self.dna["t_addestramento"]=None
        self.dna["accuracy_finale"]=None
        self.dna["fitness"]=None
        #print ("dna: ",self.dna)


    def create_individual(self):
        #selziono casualmente i geni
        self.dna["n_strati_LSTM"]=random.randint(1,4)
        self.dna["dimensione_strati_LSTM"]=random.randint(32,256)
        self.dna["n_strati_densi"]=random.randint(1,5)
        self.dna["dimensione_strati_densi"]=random.randint(32,256)
        self.dna["t_addestramento"]=None
        self.dna["accuracy_finale"]=None
        self.dna["fitness"]=None
        #print ("dna: ",self.dna)
        #print ("dna: ",self.dna)
    
    def write_on_file(self,filename):
        # Ottenere la data e l'ora correnti
        #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filename, "a") as f:
            f.write(str(self.dna)+"\n")
            f.close()
    
    def create_model(self):
        rete=NeuralNetwork(self.dna["n_strati_LSTM"],self.dna["dimensione_strati_LSTM"],
                           self.dna["n_strati_densi"],self.dna["dimensione_strati_densi"]) #creo ogg NeuralNetwork
        (x_train, y_train), (x_test, y_test) = rete.load_data() #recupero i dati 
        accuracy, training_time = rete.train(x_train, y_train) #eseguo training
        test_loss,test_accuracy = rete.evaluate(x_test, y_test) #valuto il modellp
        self.dna["t_addestramento"]=training_time 
        self.dna["accuracy_finale"]=test_accuracy
        self.dna["fitness"]=self.evaluate_fitness(training_time,test_accuracy)
        
        
    #calcola fitness con funzione 
    def evaluate_fitness(self, t_addestramento, test_accuracy):
        return test_accuracy
        #minuti=t_addestramento/60
        #return test_accuracy/math.log(minuti,3) #fitness come accuaracy/log3(minuti)
    
   
'''
if __name__ == "__main__":
    dna1 = {
    "n_strati_LSTM": 10,
    "dimensione_strati_LSTM": 10,
    "n_strati_densi": 10,
    "dimensione_strati_densi": 10,
    "t_addestramento": 10,
    "accuracy_finale": 10
    }

    dna2 = {
        "n_strati_LSTM": 5,
        "dimensione_strati_LSTM": 5,
        "n_strati_densi": 5,
        "dimensione_strati_densi": 5,
        "t_addestramento": 5,
        "accuracy_finale": 5
    }

    individuo=Individual(dna1,dna2)
'''