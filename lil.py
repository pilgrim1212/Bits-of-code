from typing import Sized

from Advance_algo import is_improvement_positive


def crossover(parent1,parent2):
    return crossover_fitness_driven_order(parent1,parent2)
def mutate(ind):
    return mutation_fitness_driven_shift(ind)
def selection(population):
    return selection_rank_with_elite(population, elite_size= 1)

#Genetic algo parameters
#set initial, maximum and minimum population size
POPULATION_SIZE = 200
MIN_POPULATION_SIZE = 50
MAX_POPULATION_SIZE = 300
#Initial and minimum crossover probability
CROSSOVER_PROBABILITY = 0.5
MIN_CROSSOVER_PROBABILITY = 0.1
#initial and mimimum mutation probability
MUTATION_PROBABILITY =  0.5
MIN_MUTATION_PROBABILITY = 0.1
#minimum and maximum generations
MAX_GENERATIONS = 100
MIN_GENERATIONS = 10

##Genetic algo flow
#data collectors
fit_avg = []
fit_best = []
impr_list = []
ev_avg = []
population_size = []
mutation_prob = []
crossover_prob = [] 

#initialisation
generation_num = 0
first_population = [generate_random(len(points)) for _ in range(POPULATION_SIZE)]
best_ind = random.choice(first_population)
population = first_population.copy

##Stop algorithm if the best fitness value does not change over 5 generations by 0.1%
while generation_num < MIN_GENERATIONS or (generation_num < MAX_GENERATIONS and is_improvement_positive(fit_best, 50, 0.001)):
    #genetic algorithm operations
    generation_num += 1
    offspring = selection(population)
    crossed_offspring = crossover_operation(offspring,crossover, CROSSOVER_PROBABILITY)
    mutated_offspring = mutation_operation(crossed_offspring, mutate,  MUTATION_PROBABILITY)
    population = mutated_offspring.copy()
    best_ind, fit_avg, fit_best = stats(population,best_ind,fit_avg,fit_best)
    ev_avg.append(average(fit_avg, 10))
    impr_rate = is_improvement_positive(fit_avg, 10, 0.001)
    impr_list.append(impr_rate)
    # If population degrades need to increase mutation and crossover probability and add two random individuals to the population
    if not impr_rate:
        MUTATION_PROBABILITY = min(MUTATION_PROBABILITY * 1.1, 1)
        CROSSOVER_PROBABILITY = min(CROSSOVER_PROBABILITY * 1.1, 1)
        if len(population) < MAX_POPULATION_SIZE:
            population = population + [generate_random(len(points)) for _ in range(2)]
    #If pop improves decrease mutation and crossover probability and remove worst individual from the group
    else:
        MUTATION_PROBABILITY = max(MUTATION_PROBABILITY * 0.9, MIN_MUTATION_PROBABILITY)
        CROSSOVER_PROBABILITY = max(CROSSOVER_PROBABILITY * 0.9, MIN_CROSSOVER_PROBABILITY)
        if len(population) > MIN_POPULATION_SIZE:
            worst_ind = min(population, key=lambda ind: ind.fitness)
            population.remove(worst_ind)
