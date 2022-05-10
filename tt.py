import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt


    
    
class Individual:

    def __init__(self, gene_list: List[float]) -> None:
        self.gene_list =gene_list
        self.fitness = func(self.gene_list[0])
    
    def get_gene(self):
        return self.gene_list[0]
    
    @classmethod
    def crossover(cls, parent1, parent2):
        child1_gene, child2_gene =crossover_blend(parent1.get_gene(), parent2.get_gene(), 1, -10, 10)
        return Individual([child1_gene]), Individual([child2_gene])

  
    @classmethod
    def mutate(cls, ind):
        mutated_gene = mutate_gaussian(ind.gen_gene(), 0, 1, -10, 10)
        return Individual([mutated_gene])
    
    @classmethod
    def select(cls, population):
        return select_tournament(population, tournament_size = 3)
    #tournament selection, subgroup selected from population 

    @classmethod
    def create_random(cls):
        return Individual([random.randrange(-1000, 1000) / 100])
def generate_random(cls):
        return Individual([random.choice([0, 1]) for _ in range(cls.period * cls.employees * 3)])        
        
def crossover(parent1, parent2):
    return crossover_fitness_driven_order(parent1, parent2)
def mutate(ind):
    return mutation_fitness_driven_shift(ind)
def selection(population):
    return selection_rank_with_elite(population, elite_size = 1)
#Genetic algorithm parameters
#We set initial, maximum and minimum population size. We define the
#boundaries for the population size:

POPULATION_SIZE = 200
MIN_POPULATION_SIZE = 50
MAX_POPULATION_SIZE = 300
#Initial and minimum crossover probability:
CROSSOVER_PROBABILITY = .5
MIN_CROSSOVER_PROBABILITY = .1
#Initial and minimum mutation probability:
MUTATION_PROBABILITY = .5
MIN_MUTATION_PROBABILITY = .1
#Minimum and maximum generations:
MAX_GENERATIONS = 10_000
MIN_GENERATIONS = 100
#Genetic algorithm flow
#Data collectors:
fit_avg = []
fit_best = []
impr_list = []
ev_avg = []
population_size = []
mutation_prob = []
crossover_prob = []
#Initialization:
generation_num = 0
first_population = [generate_random(len(points)) for _ in range(POPULATION_SIZE)]
best_ind = random.choice(first_population)
population = first_population.copy()
#We stop genetic algorithm cycle if the best fitness value does not
#differ from the average over 50 generations by 0.1%:
while generation_num < MIN_GENERATIONS or (generation_num < MAX_GENERATIONS and is_improvement_positive(fit_best, 50, .001)):
    #Genetic algorithm operations:
    generation_num += 1
    offspring = selection(population)
    crossed_offspring = crossover_operation(offspring,
    crossover, CROSSOVER_PROBABILITY)
    mutated_offspring = mutation_operation(crossed_offspring, mutate,MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    best_ind, fit_avg, fit_best = stats(population,
    best_ind, fit_avg, fit_best)
    ev_avg.append(average(fit_avg, 10))
    impr_rate = is_improvement_positive(fit_avg, 10, .001)
    impr_list.append(impr_rate)
    #If the population degrades, we increase mutation and crossover
    #probability, and add two random individuals to the population:
    if not impr_rate:
        MUTATION_PROBABILITY = min(MUTATION_PROBABILITY * 1.1,1)
        CROSSOVER_PROBABILITY = min(CROSSOVER_PROBABILITY *1.1, 1)
        if len(population) < MAX_POPULATION_SIZE:
            population = population + [generate_random(len(points)) for _ in range(2)]
        #And if the population improves, we decrease mutation and
        #crossover probability, and remove the worst individual from the
        #population:
    else:
        MUTATION_PROBABILITY = max(MUTATION_PROBABILITY * .99,MIN_MUTATION_PROBABILITY)
        CROSSOVER_PROBABILITY = max(CROSSOVER_PROBABILITY *.99, MIN_CROSSOVER_PROBABILITY)
        if len(population) > MIN_POPULATION_SIZE:
            worst_ind = min(population, key = lambda ind: ind.fitness)
            population.remove(worst_ind)
