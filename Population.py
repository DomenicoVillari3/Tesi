import random
from Individual import *

file_name='individui.txt'

class Population:
    #mutation rate pari al 10%
    mutation_rate=.1

    def __init__(self,size,file=None,resume=False):
        #definisco attributi classe : dimensione, popolazione, selezionati, e progenie
        self.size=size
        self.individuals=[]
        self.selected=[]
        self.offspring=[]

        #se non viene passato il file sono generati tutti i modelli
        # altrimenti viene creata la popolazione sulla base del file
        if file==None:
            #print("if")
            for i in range(size) :
                individuo=Individual()
                individuo.create_model()
                individuo.write_on_file(file_name)
                self.individuals.append(individuo)

        #se devo recuperare da file senza resume recupero i primi size individui dal file
        elif file!=None and resume==False:
            #print("else")
            i=0 #conta individui inseriti

            with open(file, "r") as f:
                        
                    for line in f:
                        individuo=Individual(eval(line))
                        self.individuals.append(individuo)
                        i+=1
                        if i==size:
                            break
                    
                    #se nel file non ci sono abbastanza individui entro nel while
                    while i<size:
                        individuo=Individual()
                        self.individuals.append(individuo)
                        individuo.create_model()
                        individuo.write_on_file(file_name)
                        i+=1
                    
                    f.close()

        #recupero da file gli ultimi size individui per ripartire da una generazione successiva 
        elif file!=None and resume==True:    
            ultima_gen=[]#lista per recuperare gli individui 

            with open(file, "r") as f: 

                individui_all=f.readlines()#ritorna una lista con tutte le righe
                
                #metto nella lista gli individui dell'ultima gen, gli ultimi size elementi
                ultima_gen = individui_all[-size:]
                
                for individuo in ultima_gen:
                     self.individuals.append(Individual(eval(individuo)))
                
                f.close()


                    
    # Funzione per normalizzare le fitness sulla generazione corrente
    def normalize_fitness(self):
    
        smallest_fitness=self.individuals[0].dna["fitness"]
        largest_fitness=self.individuals[0].dna["fitness"]

        #recupero le fitness più grandi e più piccola dell'array individuals
        for i in self.individuals:
            if i.dna["fitness"]<smallest_fitness:
                smallest_fitness=i.dna["fitness"]
            if i.dna["fitness"]>largest_fitness:
                largest_fitness=i.dna["fitness"]

        for i in self.individuals:
            i.normalized_fitness = (i.dna["fitness"]-smallest_fitness) / (largest_fitness-smallest_fitness)
            #print("NORMALIZZATA: ",i.normalized_fitness)
    

    #funzione per selezionare gli individui
    def select(self):
        self.normalize_fitness() #definisco la fitness normalizzata

        selecting=True
        self.selected.clear()
        
        while selecting:
            for i in self.individuals:
                #se si ha un valore random < della fitness normalizzata dell'individuo si seleziona
                if random.random() < i.normalized_fitness :
                    #print("selezione ",j)
                    #j+=1
                    #print("acc=",i.dna["accuracy_finale"]," r=",r)
                    self.selected.append(i)

                #lo si fa finchè non si raggiunge la size della popolazione predefinita
                if len(self.selected) == self.size:
                    selecting=False
                    break


    #funzione per generare le progenie
    def crossover(self):
        self.offspring.clear()  #pulisco la lista dei figli per essere sicuro di avere size individui

        for i in range(self.size):
            i1=None
            i2=None
            #si prendono 2 indidui random
            i1=random.randint(0,self.size-1)
            i2=random.randint(0,self.size-1)
            #si appende alla progenie l'individuo figlio dei 2 
            individuo=Individual(self.selected[i1].dna,self.selected[i2].dna)
            self.offspring.append(individuo)


    #funzione per mutare le progenie
    def mutate(self):
        #per ogni individuo nell'offspring si muta 1 valore con una probabilità del 10%
        for i in self.offspring:    
            if random.random() < self.mutation_rate:
                print("PRE-MUTAZIONE: ",i.dna)
                attribute,new_value=self.random_modify() #funzione che cambia un valore
                i.dna[attribute]=new_value
                print("POST-MUTAZIONE: ",i.dna)
                print("\n")

          
                    
    def addestra_offspring(self):
        for individuo in self.offspring:
            print(individuo.dna)
            individuo.create_model()
            individuo.write_on_file(file_name)
            
          
    
    
    #funzione per ottenere l'individuo con la miglior fitness
    def get_best_Individual(self):
        bestIndividual=self.individuals[0]
        bi_fitness=bestIndividual.dna["fitness"]

        for i in self.individuals:
            if i.dna["fitness"] > bi_fitness:
                bestIndividual=i
                bi_fitness=i.dna["fitness"]
        return bestIndividual
    

    #funzione per scegliere un valore da modificare nel dna dell'individuo    
    def random_modify(self):
        #prendo le chiavi del dizionario del dna di cui si vuole modificare il valore con la mutazione
        individual_attributes=["n_strati_convoluzionali","dimensione_strati_convoluzionali","n_strati_densi","dimensione_strati_densi"]
        #seleziono un valore da modificare nel dizionario casualmente e poi ritorno il campo da modificare col nuovo valore
        target=random.randint(0,len(individual_attributes)-1)

        if(target==0):  #n_strati_convoluzionali
            new_value=random.randint(1,4)
            
        elif(target==1):    #dimensione_strati_convoluzionale
            new_value=random.randint(32,256)
           
        elif(target==2):    #n_strati_densi
            new_value=random.randint(1,5)
           
        elif(target==3):    #dimensione_strati_densi
            new_value=random.randint(32,256)
            
        return individual_attributes[target], new_value

'''
if __name__=="__main__":
    pop=Population(20,file="AI\progetto\generazione 20\individui20.txt")
    pop.select()
    print(pop.selected)
    pop.crossover()
    pop.mutate()
    print("miglior individuo: ",pop.get_best_Individual().dna)
    print("---------------------------------------------------------------------------------------------")
    print("progenie: \n")
    for i in pop.offspring:
        print(i.dna)
'''