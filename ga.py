import sys

import numpy as np

def fitness(equationInputs, population):

    soma = np.sum(population * equationInputs, axis=1)
    fitness = []
    for i in soma :
        if (i > 30 ) :
            i *= -9999999
        fitness.append(i)
    return np.array(fitness)
def selection(population, fitness, numberParents):

    parents = np.empty((numberParents, population.shape[1])) # 4 x 6

    for idx in range(numberParents):

        maxFitnessIdx = np.where(fitness == np.max(fitness))
        maxFitnessIdx = maxFitnessIdx[0][0]
        parents[idx, :] = population[maxFitnessIdx, :]
        fitness[maxFitnessIdx] = -999999
    return parents
def crossover(parents, generationSize):

    offspring = np.empty(generationSize)
    
    crossoverPoint = np.uint8(generationSize[1]/2)
    
    for idx in range(generationSize[0]):

        p1Idx = idx % parents.shape[0]
        
        p2Idx = (idx + 1) % parents.shape[0]

        offspring[idx, 0:crossoverPoint] = parents[
            p1Idx, 0:crossoverPoint]

        offspring[idx, crossoverPoint:] = parents[
            p2Idx, crossoverPoint:
        ]
    return offspring
def mutation(offspring):
    for idx in range(offspring.shape[0]):
        
        randomIdx = np.random.randint(offspring.shape[1])

        offspring[idx, randomIdx] = (
            abs(offspring[idx, randomIdx] - 1)
        )
    return offspring