from NeuralNetwork import *
from Individual import *
from Population import *
import csv
import os



def write_on_file(gen,individui,filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
            
            # Definisce i nomi delle colonne nel file CSV
            fieldnames = ['generazione', 'dna','tempo','accuracy', 'fitness',"normalized_fitness"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            #scrivo intestazione
            if not file_exists:
                writer.writeheader()

            for individuo in individui:
                # Estrae i geni dall'individuo corrente e li  metto in una lista
                geni=[individuo.dna["n_strati_convoluzionali"],individuo.dna["dimensione_strati_convoluzionali"],
                      individuo.dna["n_strati_densi"],individuo.dna["dimensione_strati_densi"]]
                
                #scrivo su file
                writer.writerow({'generazione': gen, 'dna': geni,
                                 'tempo':individuo.dna["t_addestramento"],'accuracy':individuo.dna["accuracy_finale"],
                                  'fitness': individuo.dna["fitness"],"normalized_fitness": individuo.normalized_fitness})
                geni.clear()
            file.close()


def main():
    running=True #flag per controllare se l'algoritmo ha terminato
    generation=1 #numero di generazione corrente VA CAMBIATO SE VA FATTO IL RESUME 
    pop=Population(size=20) #popolazione iniziale di 20 individui

    while running:
        print("Generazione: ", generation,"con ",len(pop.individuals)," individui")
        pop.select()    #seleziono gli individui

        write_on_file(generation,pop.individuals,"AI\progetto\Generazione.csv") #scrivo su file.csv la generazione corrente

        print("crossover su generazione", generation) 
        pop.crossover() #fase di crossover
        print("mutation su generazione", generation)    
        pop.mutate() #fase di mutazione 
        pop.addestra_offspring()
        pop.individuals=pop.offspring #rendo i figli della popolazione attuale la nuova popolazione
        
        i=pop.get_best_Individual() #recupero l'individuo che ha la miglior accuracy
        print("BEST INDIVIDUAL: ",i.dna)
        
        if i.dna["fitness"]>=0.91: #controlla se l'individuo ha un accuracy soddisfacente
            running=False  #interrompo l'esecuzione 
            print("BEST INDIVIDUAL TROVATO : ",i.dna)
            #write_on_file(generation,pop.individuals,"AI\progetto\Generazione.csv") #messo per scrivere anche quando trova l'individu0
        
        generation+=1
    
       

if __name__ == "__main__":
    main()