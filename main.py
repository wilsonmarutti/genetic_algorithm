import numpy as np
import ga

def main():
    equationInputs = [15, 10, 5, 5, 8, 17] 

    numberWeights = 6  
    solutionsPerPopulations = 6  
    
    populationsSize = (solutionsPerPopulations, numberWeights)
    
    population = np.random.randint(
        low=0, high=2,
        size=populationsSize)
    print("População inicial:")
    print(population)

    numberGenerations = 5

    numberParentsCrossover = 4
    
    for generation in range(numberGenerations):
        print(f"\nGeração {generation}")
        
        fitness = ga.fitness(equationInputs, population)
        print("\nFitness:")
        print(fitness)
        
        selectedParents = ga.selection(
            population, fitness, numberParentsCrossover)
        print("\nGenitores selecionados:")
        print(selectedParents)
        
        offspringCrossover = ga.crossover(
            selectedParents, (
                solutionsPerPopulations - numberParentsCrossover,
                numberWeights
            )
        )
        print("\nFilhos gerados por crossover:")
        print(offspringCrossover)

        offspringMutation = ga.mutation(offspringCrossover)
        print("\nFilhos pós mutação:")
        print(offspringMutation)

        population[0:selectedParents.shape[0], :] = selectedParents

        population[selectedParents.shape[0]:, :] = offspringMutation
        print("\nNova população:")
        print(population)
        print("Melhor resultado: ", np.max(
            ga.fitness(equationInputs, population)))
    fitness = ga.fitness(equationInputs, population)
    bestFitIdx = np.where(fitness == np.max(fitness))
    print("Melhor resultado: ", population[bestFitIdx, :])
    print("Fitness do melhor: ", fitness[bestFitIdx])
if __name__ == "__main__":
    main()