import sys
import numpy as np
from random import SystemRandom
import matplotlib.pyplot as plt
from tqdm import tqdm

K = 0.5
crossover_probability = 0.75

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

def check_constraints(vector, limits, constraint_func):
    x = vector[0]
    y = vector[1]
    if limits.x_min <= x <= limits.x_max and limits.y_min <= y <= limits.y_max and constraint_func(x, y):
        return True
    return False

def crossover(mutant, candidate):
    trial = [mutant[i] if SystemRandom().random() <= crossover_probability else candidate[i] for i in range(len(candidate))]
    return trial
# # trial = [None for i in range(len(mutant))]
#     trial = []
#     for i in range(len(candidate)):
#         crossover_point = SystemRandom().random()
#         if crossover_point <= crossover_probability:
#             # trial[i] = mutant[i]
#             trial.append(mutant[i])
#         else:
#             # trial[i] = candidate[i]
#             trial.append(candidate[i]) 
#     return trial

def dominate(v1, v2, functions):
    flag = 0
    for f in functions:
        v1_fitness = f(v1)
        v2_fitness = f(v2)
        if v1_fitness > v2_fitness: return False
        if v1_fitness < v2_fitness: flag = 1
    if flag == 1: return True
    return False

def get_front(population, functions):
    # Initialize n and S
    n = [0] * len(population)
    S = []
    
    # Pareto dominance to choose front members
    for index, m in enumerate(population):
        Si = []
        for other_m in population:
            if (dominate(m, other_m, functions)):
                Si.append(index)
            elif dominate(other_m, m, functions): 
                n[index] += 1
        S.append(Si)

    front = []
    front_indices = []
    # Find front members
    for index, ni_value in enumerate(n):
        if (ni_value==0):
            front.append(population[index])
            front_indices.append(index)
    
    for index, member in enumerate(front):
        # j represents indices of members that this member dominates
        for j in S[front_indices[index]]:
            n[j] -= 1

    # new_population = [member for member in population if member not in front]
    new_population = [member for i, member in enumerate(population) if i not in front_indices]

    if (len(front) == 0):
        exit()
    return front, new_population

def crowding_distance(population, functions):
    crowd_dist = [sys.maxsize] * len(population)

    # For each function
    for function in functions:
        # Sort population in ascending order of function values
        func_values = []
        for member in population:
            func_values.append(function(member))
        func_values.sort()
        sorted_pop = [member for func_value, member in sorted(zip(func_values, population), key=lambda x: x[0])]

    # For each member in sorted population, find crowding distance
    for index in range(1, len(sorted_pop)-1):
        crowd_dist[index] += (np.abs(function(sorted_pop[index-1]) - function(sorted_pop[index+1]))/np.abs(function(sorted_pop[0]) - function(sorted_pop[len(sorted_pop)-1])))

    # Sort population in descending order of crowding distances
    final_sorted_pop = [member for dist, member in sorted(zip(crowd_dist, sorted_pop), key=lambda x: x[0], reverse=True)]
    return final_sorted_pop

def nsde(population, functions):
    size = len(population)/2 # Size of new generation
    next_gen = []

    unfilled_spots = size # N
    while(len(next_gen) < size):
        # Generate a front
        # New population = Population - front members
        front, new_population = get_front(population, functions)

        # If size of front is smaller than N, add all members
        if (len(front) <= unfilled_spots):
            for member in front: 
                next_gen.append(member)
            unfilled_spots -= len(front)

        # If size of front is greater than N, choose members using crowding distance algo
        else:
            result = crowding_distance(population, functions)[:int(unfilled_spots)]
            for member in result:
                next_gen.append(member)
            unfilled_spots -= len(result)
        population = new_population[:]
    return next_gen

def plot_2_obj(population, func_name, functions):
    x_vals = [functions[0](candidate) for candidate in population]
    y_vals = [functions[1](candidate) for candidate in population]
    plt.scatter(x_vals, y_vals)
    plt.xlabel("Function 1")
    plt.ylabel("Function 2")
    plt.title("{func_name} function pareto front".format(func_name = func_name))
    plt.show()

def moode(pop_size, gens, func_name, functions, lims, constraint_funcs):
    limits = set_limits(lims)
    
    parents = initialize_candidates(pop_size, limits)
    print("Generations:")
    for g in tqdm(range(gens)):
        trials = []
        F = get_F()
        for index, candidate in enumerate(parents):
            while(True):
                mutant = mutation(parents, index, K, F, candidate)
                trial = crossover(mutant, candidate)
                for cf in constraint_funcs:
                    if (check_constraints(trial, limits, cf) == False): continue
                break        
            trials.append(np.array(trial))

        # NSDE selection
        next_gen = nsde(parents+trials, functions)
        parents = next_gen[:]
    # Plotting
    plot_2_obj(next_gen, func_name, functions)