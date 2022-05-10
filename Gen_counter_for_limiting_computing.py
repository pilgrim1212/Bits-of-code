from random import randint, random, uniform
from typing import MutableSet
import numpy as np
import sympy as sp
import matplotlib as plt
##system of equations
def F(x,y,z):
    return (x*y+2)**2 + (x+y) **(abs(z-2)**3+2) + x*y*z

def g(x,y,z):
    return x* (y*z+10) - factorial(abs(z-3)) + y**abs(x) + 11 * z

def w(x,y,z):
    return (x + 7 * y)**abs(z+x) - (z + 16)**2  - 151

##Fitness function
def F(x,y,z):
    return abs(F(x,y,z)) + abs(g(x,y,z) + abs(w(x,y,z)))

##Individual
class Individual:
    def __init__(self, gene_list) -> None:
        self.gene_list = [constraints(g) for g in gene_list]        
        self.fitness = -F(self.gene_list[0], self.gene_list[1], self.gene_list[2])
    
    def __str__(self):
        return f'x: {self.gene_list[0]}, y: {self.gene_list[1]}, z: {self.gene_list[2]}, fitness: {self.fitness} '


        ##Genetic algorithm structure
    def crossever(parent1,parent2):
        child1_genes, child2_genes =crossever_blend(parent1.gene_list, parent2.gene_list, 0.8)
        return Individual(child1_genes), Individual(child2_genes)


    def mutate(ind):
        mutated_gene = mutate_random_deviation(ind.gene_list, 0, 3, 0.5)
        return Individual(mutated_gene)

    def select(population):
        return selection_rank_with_elite(population, elite_size = 2)

    def create_random():
        return Individual([round(random.uniform(-10,10,), 2), (round.uniform(-10,10), 2), (round.uniform(-10,10), 2 )])

##Genetic algorithm Parameters
POPULATION_SIZE = 400
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.4
MAX_GENERATIONS = 100


##Genetic algorithm flow
## Genetic algorithm flow stop if the fitness value is 0
first_population = [create_random() for _ in range(POPULATION_SIZE)]
best_ind = random.choice(first_population)
#Allows us to store the generation number so it can be called later to prevent it going over the maximum generation limit that is set
generation_number = 0
population = first_population.copy()
while generation_number < MAX_GENERATIONS and best_ind.fitness != 0:
    generation_number += 1 #update the generation number stored every iteration
    offspring = select(population)
    crossed_offspring = crossover_operation(offspring, crossover, CROSSOVER_PROBABILITY)
    mutated_offspring = mutation_operation(crossed_offspring, mutate, MUTATE_PROBABILITY)
    population =mutated_offspring.copy()
    best_of_generation = max(population, key = lambda ind: ind.fitness)
    if best_ind.fitness < best_of_generation.fitness:
        best_ind = best_of_generation
    print(f'Generation: {generation_number}, bestfit: {best_ind.fitness}')


