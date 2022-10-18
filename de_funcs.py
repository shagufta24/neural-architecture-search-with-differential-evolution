import sys
import numpy as np
from random import SystemRandom
import matplotlib.pyplot as plt

K = 0.5
crossover_probability = 0.5
graph_store = {}

class Generation(object):
    def __init__(self, best_candidate, best_fitness, average_fitness):
        self.best_candidate = best_candidate
        self.best_fitness = best_fitness
        self.average_fitness = average_fitness

class set_limits(object):
    def __init__(self, lims):
        self.x_min = lims[0]
        self.x_max = lims[1]
        self.y_min = lims[2]
        self.y_max = lims[3]

def initialize_candidates(n, limits):
    initial_vectors = []
    for i in range(n):
        rand_x = SystemRandom().uniform(limits.x_min, limits.x_max)
        rand_y = SystemRandom().uniform(limits.y_min, limits.y_max)
        initial_vectors.append(np.array([rand_x, rand_y]))
    return initial_vectors

def get_F(_min=-2, _max=2):
    return SystemRandom().uniform(_min, _max + sys.float_info.epsilon)

def mutation(parents, index, K, F, candidate):
    parents_minus_i = parents[:]
    parents_minus_i.pop(index)
    r1,r2,r3 = SystemRandom().sample(parents_minus_i, 3)
    mutant = candidate + K*(r1 - candidate) + F*(r2 - r3)
    return mutant

# ISSUE - need to add contraints in moode
def check_constraints(vector, limits, constraint_func):
    x = vector[0]
    y = vector[1]
    if limits.x_min <= x <= limits.x_max and limits.y_min <= y <= limits.y_max:
        if (constraint_func(x, y) == True):
            return True
        else:
            return False
    else:
        return False

def crossover(mutant, candidate):
    trial = [None for i in range(len(mutant))]

    for i in range(len(trial)):
        crossover_point = SystemRandom().random()
        if crossover_point <= crossover_probability:
            trial[i] = mutant[i]
        else:
            trial[i] = candidate[i]  
    return trial

def elitism(candidate, trial, function):
    candidate_fitness = function(candidate)
    trial_fitness = function(trial)

    if candidate_fitness < trial_fitness:
        return candidate
    else:
        return trial

def global_comp(vectors, function):
    best_fitness = sys.maxsize
    average_fitness = 0
    for candidate in vectors:
        fitness = function(candidate)
        average_fitness += fitness
        if fitness < best_fitness:
            best_fitness = fitness
            best_candidate = candidate
    average_fitness /= len(vectors)
    return Generation(best_candidate, best_fitness, average_fitness)

def plot(function_name, best_of_all_gen, pop_size, gens):
    x = [i for i in range(len(best_of_all_gen))]
    y_best = [i.best_fitness for i in best_of_all_gen]
    y_avg = [i.average_fitness for i in best_of_all_gen] 
    
    plt.suptitle(function_name+"#Generations: {}".format(gens), fontsize=16,y=2.09)
    plt.title("Population size: {}".format(pop_size))
    plt.plot(x, y_avg, label='Average value')
    plt.legend()
    plt.plot(x, y_best, label='Best value')
    plt.legend()
    plt.xlabel('Number of generations -- >')
    plt.ylabel('Function value -->')
    plt.legend()
    plt.show()

def de(pop_size, gens, function, lims, constraint_func):
    limits = set_limits(lims)
    best_of_all_gen = []
    
    parents = initialize_candidates(pop_size, limits)
    for g in range(gens):
        selected = []
        # print("Generation {gen}:\n\n".format(gen = g))
        F = get_F()
        for index, candidate in enumerate(parents):
            while(True):
                # print("Candidate {cand_no}: {cand}\n".format(cand_no=index, cand=candidate))
                mutant = mutation(parents, index, K, F, candidate)
                # print("Mutant: {mut}\n".format(mut=mutant))
                trial = crossover(mutant, candidate)
                chosen_vector = elitism(candidate, trial, function)
                if (check_constraints(chosen_vector, limits, constraint_func) == True):
                    break
            selected.append(np.array(chosen_vector))

        best_of_gen = global_comp(selected, function)
        best_of_all_gen.append(best_of_gen)
        parents = selected[:]
    # print ('Function: {}\n#Gens : {}\nPop size : {}\nBest fitness : {}\nBest candidate : {}\n'.format(function.__name__.title(), gens, pop_size, best_of_gen.best_fitness, best_of_gen.best_candidate))
    plot(function.__name__.title(), best_of_all_gen, pop_size, gens)


