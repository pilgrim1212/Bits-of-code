# Genetic Algorithm to optimize polynomial
import numpy as np
import tabulate
import random

def fitness_score(arr):
    f=[]
    integers=[]
    for x in arr:
        s=np.sum(2**np.arange(len(x))*x)#binary to decimal
        integers.append(s)
        f.append(2*s-(s*s/16))
    return f,integers

def takefifth(elem):
    return elem[5]
 
def deterministic_sampling(fitness,population,population_size):
    g=[]
    sorted_population=[]
    for i in range(population_size):
        normalize=population_size*(fitness[i]/sum(fitness))
        x=np.append(population[i],[normalize])
        sorted_population.append(x)
        for j in range(int(normalize)):
            g.append(list(population[i][0:5]))
    sorted_population=sorted(sorted_population,key=takefifth)
    if(len(g)!=population_size):
        for i in range(population_size):
            if(sorted_population[i][5]<1):
                g.append(list(sorted_population[i][0:5].astype(int)))
            if(len(g)==population_size):
                break
    return g,sorted_population

def crossover(initial_population,population_size):
    c=[]
    new_population=[]
    mates=initial_population[:]
    np.random.shuffle(mates)
    for i in range(population_size):
        r = random.randint(0, 5)
        c.append(r)
        if(i%2==0):
            temp=initial_population[i][0:r]
            temp.extend(list(mates[i][r:]))
        else:
            temp=mates[i][0:r]
            temp.extend(list(initial_population[i][r:]))
        new_population.append(temp)
    header = ['Initial Population', 'Mates','Crossover site','New Population']
    print(tabulate.tabulate([[initial_population[i][::-1],mates[i][::-1],c[i],new_population[i][::-1]] for i in range(population_size)], headers=header, tablefmt='grid'))
    return new_population

def mutation(population,populationsize):
    r= random.randint(0, 4)
    i= random.randint(0, 3)
    print(r,i)
    if (population[i][r]==0):
        population[i][r]=1
    else:
        population[i][r]=0
    return population

def genetic():
    population_size=4
    #generating initial population of 32 chromosomes(integers)
    chromosomes=np.arange(32)
    #converting to binary
    chromosomes_bin= (((chromosomes[:,None] & (1 << np.arange(5))))>0).astype(int)
    print("Chromosomes\n",chromosomes_bin)
    
    #selecting initial population(replace=False to avoid repititions)
    random_rows=np.random.choice(chromosomes_bin.shape[0],population_size,replace=False)
    initial_population=chromosomes_bin[random_rows,:]
    
    #calculate fitness scores of initial population
    fitness_scores,integers=fitness_score(initial_population)

    #deterministic sampling for parent selection
    #chromosomes with low fitness discarded
    #chromosomes with high fitness repeated
    parents,sorted_population=deterministic_sampling(fitness_scores,initial_population,population_size)
 
    header = ['Population', 'Integers','Fitness Score','Parents for next generation']
    print(tabulate.tabulate([[initial_population[i][::-1],integers[i],fitness_scores[i],parents[i][::-1]] for i in range(population_size)], headers=header, tablefmt='grid'))

    def sublists_equal(a, b):
        return all(l for l in b if l in a)

    while(max(fitness_scores)!=16):
        initial_population=parents[:]
        #generate new population using simple crossover method
        new_population=crossover(initial_population,population_size)
        
        #calculate fitness score
        fitness_scores,integers=fitness_score(new_population)

        #deterministic sampling for parent selection
        #chromosomes with low fitness discarded
        #chromosomes with high fitness repeated
        parents,sorted_population=deterministic_sampling(fitness_scores,new_population,population_size)
        parents=mutation(parents,population_size)

        header = ['Population', 'Integers','Fitness Score','Parents for next generation']
        print(tabulate.tabulate([[new_population[i][::-1],integers[i],fitness_scores[i],parents[i][::-1]] for i in range(population_size)], headers=header, tablefmt='grid'))
    
        

genetic()