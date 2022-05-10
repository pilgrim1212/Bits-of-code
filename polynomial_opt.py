from numpy.lib.function_base import corrcoef
import pandas as pd
import numpy as np
## create a dataframe
df = pd.DataFrame(np.arange(-2.2, 2.2, 0.2), columns = ['x'])
df['square'] = df['x']**2
##Create the lookup table
lookupTable= {}
for i, record in df.iterrows():
    key = record['x']
    lookupTable[key] = record['square']

##Coding the genetic algorithm
import numpy.random as rand
from copy import deepcopy
import matplotlib.pyplot as plt

class Individual:
    #c is the number of coefficients 
    #d is the number of variables   
    def __init__(self, c, d):
        #generate normal distributed coefficients for each variable
        self.values =   [[rand.normal() for _ in range (c+1)] for _ in range(d)]
        self.fitness = None
    ##Now have to evaluate the preformance/fitness of the individual, this case
    #how well it fits the points
    def evaluate(self,lookupTable):
        self.fitness = 0
        #For each input
        for x in lookupTable.keys():
            image = 0
            #for each variable
            for variable in self.values:
                #For each coefficient
                for power, coefficient in enumerate(variable):
                    #Compute polynomial image
                    image += coefficient * x ** power
                #Compute squared error
                target = lookupTable[x]
                mse = (target - image) ** 2
                self.fitness += mse
    def mutate(self, rate):
        #Coeffients take a random value in their neighbourhood
        self.values =[[rand.uniform(c - rate, c + rate) for c in variable] for variable in self.values]

    def display(self):
        intercept = 0
        print('Polynomial form')
        print('---------------')
        for index, variable in enumerate(self.values):
            intercept += variable[0]
            for power, coefficient in enumerate(variable[1:]):
                print(str(coefficient) + ' * ' + 'x' + \
                          str(index) + '**' + str(power+1) + ' + ')
            print(intercept)

class population:

    def __init__(self, c, d, size=100):
        #Create individuals
        self.individuals = [Individual(c,d) for _ in range(size)]
        #Store the best individuals
        self.best = Individual[(c, d)]
        #Mutation rate
        self.rate = 0.1
    def sort(self):
        self.individuals = sorted(self.individuals, key = lambda indi: indi.fitness)
    def evaluate(self, lookupTable):
        for indi in self.individuals:
            indi.evaluate(lookupTable)
        self.sort()
def enhance(self, lookupTable):
    newIndividuals = []
    #Go through the top 10 individuals
    for individual in self.individuals[:10]:
        #Copy exactly one of each of the top ten individuals 
        newIndividuals.append(deepcopy(individual))
        #Create 4 mutated individuals
        for _ in range(4):
            newIndividual = deepcopy(individual)
            newIndividual.mutate(self.rate)
            newIndividuals.append(newIndividual)
        #Replace the old poputlation with the new population
        self.individuals = newIndividuals
        self.evaluate(lookupTable)
        self.sort()
        #Store the best individual
        self.best.append(self.individuals[0])
        #Increment the mutation rate if the population didnt change
        if self.best[-1].fitness == self.best[-2].fitness:
            self.rate += 0.01
        else:
            self.rate = 0.1

generations = 300
degrees = 2
variables = 1

polynomials = population(degrees, variables)
polynomials.evaluate(lookupTable)
polynomials.sort()

for g in range(generations):
    #Enhance the population
    polynomials.enhance(lookupTable)

polynomials.best[-1].display()

