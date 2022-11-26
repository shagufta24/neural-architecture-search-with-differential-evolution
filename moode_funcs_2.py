import sys
import numpy as np
from random import SystemRandom
import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)

K = 0.5
crossover_probability = 0.75

class set_limits(object):
    def __init__(self, lims):
        self.io_min = lims[0]
        self.io_max = lims[1]
        self.latent_min = lims[2]
        self.latent_max = lims[3]

def initialize_mapping(n_inputs):
    global rand_nums
    rand_nums = np.array([SystemRandom().uniform(0, 1) for i in range(2*n_inputs)])
    rand_nums.sort()
    global mapping
    mapping = {rand_nums[i]: i for i in range(2*n_inputs)}

# Cadidate: An autoencoder neural network
# Structure: Input layers: 1, 2, 3, 4, Latent layer, Output layers: 5, 6, 7, 8
# [layer1_nodes, layer2_nodes, layer3_nodes, layer4_nodes, layer5_nodes, layer6_nodes, layer7_nodes, layer8_nodes, latent_layer_nodes]

def initialize_candidates(cand_size, pop_size, limits):
    initial_vectors = []
    cand = []
    for i in range(pop_size):
        cand = np.array([SystemRandom().uniform(0, 1) for j in range(cand_size)])
        initial_vectors.append(cand)
    return initial_vectors

def get_F(_min=-2, _max=2):
    return SystemRandom().uniform(_min, _max + sys.float_info.epsilon)

def mutation(parents, index, K, F, candidate):
    parents_minus_i = parents[:]
    parents_minus_i.pop(index)
    r1,r2,r3 = SystemRandom().sample(parents_minus_i, 3)
    mutant = candidate + K*(r1 - candidate) + F*(r2 - r3)
    return mutant

def check_constraints(candidate, limits):
    for x in candidate[:-1]:
        if not(limits.io_min <= x <= limits.io_max):
            return False
    if not(limits.latent_min <= candidate[-1] <= limits.latent_max):
        return False
    return True

def crossover(mutant, candidate):
    rand = SystemRandom().random()
    trial = [mutant[i] if rand <= crossover_probability else candidate[i] for i in range(len(candidate))]
    return trial

def float_to_int_mapping(vector):
    mapped = []
    for x in vector:
        closest_id = (np.abs(rand_nums - x)).argmin()
        mapped.append(rand_nums[closest_id])
    int_vector = []
    for i in mapped:
        int_vector.append(mapping[i])
    return int_vector

def dominate(v1, v2, functions):
    flag = 0
    for f in functions:
        v1_fitness = f(v1)
        v2_fitness = f(v2)
        if v1_fitness > v2_fitness: return False
        if v1_fitness < v2_fitness: flag = 1
    if flag == 1: return True
    return False

def get_front(population, mapped_pop, functions):
    # Initialize n and S
    n = [0] * len(population)
    S = []

    # Pareto dominance to choose front members
    # Computing ni and Si for each member
    for index, member in enumerate(mapped_pop):
        Si = []
        for other_member in mapped_pop:
            if (dominate(member, other_member, functions)):
                Si.append(index)
            elif dominate(other_member, member, functions): 
                n[index] += 1
        S.append(Si)
    # print("Pareto front: ni and Si computed\n")

    # Find front members. If ni = 0, add to front
    front = []
    mapped_front = []
    front_indices = []
    for index, ni_value in enumerate(n):
        if (ni_value==0):
            front.append(population[index])
            mapped_front.append(mapped_pop[index])
            front_indices.append(index)
    
    # Reduce nj by 1 for each member j belonging to Si of a front member i
    for index, member in enumerate(front):
        # j represents indices of members that this member dominates
        for j in S[front_indices[index]]:
            n[j] -= 1
    
    # print('Found front, population updated\n')
    # Remove the front members from the population
    # new_population = [member for member in population if member not in front]
    new_population = [member for i, member in enumerate(population) if i not in front_indices]
    new_mapped_pop = [member for i, member in enumerate(mapped_pop) if i not in front_indices]

    # If front is empty, there are no non-dominating members in population
    if (len(front) == 0):
        exit()
    return front, mapped_front, new_population, new_mapped_pop

def crowding_distance(population, mapped_pop, functions):
    # Array of Di for each member i of front k
    crowd_dist = [sys.maxsize] * len(population)

    # For each function
    for f in functions:
        # Sort population in ascending order of objective function values
        func_values = []
        for member in mapped_pop:
            func_values.append(f(member))
        
        population_list = [member.tolist() for member in population]
        sorted_pop = [np.array(member) for _, member in sorted(zip(func_values, population_list))]
        sorted_mapped_pop = [member for _, member in sorted(zip(func_values, mapped_pop))]

        # print('Population sorted by function values\n')
    
        # For each member in sorted population, find crowding distance di
        # Add computed di to Di
        for index in range(1, len(sorted_mapped_pop)-1):
            f_prev = f(sorted_mapped_pop[index-1])
            f_next = f(sorted_mapped_pop[index+1])
            f_first = f(sorted_mapped_pop[0])
            f_last = f(sorted_mapped_pop[len(sorted_pop)-1])
            crowd_dist[index] += (np.abs(f_prev - f_next)/(np.abs(f_first - f_last)+sys.float_info.epsilon))
        # print('Crowding dist computed\n')

    # Sort population in descending order of crowding distances
    final_sorted_pop = [member for dist, member in sorted(zip(crowd_dist, sorted_pop), key=lambda x: x[0], reverse=True)]
    return final_sorted_pop

def nsde(population, functions):
    size = len(population)/2 # Size of new generation
    next_gen = []

    # Map all candidates from float to integer space
    mapped_pop = []
    for vector in population:
        mapped_pop.append(float_to_int_mapping(vector))
    # print('Population mapped to integer space\n')

    unfilled_spots = size # N
    while(len(next_gen) < size):
        # Generate a front
        # New population = Population - front members
        front, mapped_front, new_population, new_mapped_pop = get_front(population, mapped_pop, functions)

        # If size of front is smaller than the remaining spots, add all front members to next gen
        if (len(front) <= unfilled_spots):
            for member in front:
                next_gen.append(member)
            unfilled_spots -= len(front)
            # print('Added all members of front to next gen\n')

        # If size of front is greater than the remaining spots, choose members using crowding distance algo
        else:
            # print('Crowding distance algo\n')
            result = crowding_distance(front, mapped_front, functions)[:int(unfilled_spots)]
            for member in result:
                next_gen.append(member)
            unfilled_spots -= len(result)

        # print('NSDE done\n')
        population = new_population[:]
        mapped_pop = new_mapped_pop[:]
    return next_gen

def moode(pop_size, cand_size, n_inputs, gens, functions):
    retry_count = 0
    # Create the random number mapping to map from float to int space
    initialize_mapping(n_inputs)

    # Set the constraints for each gene
    for key, value in mapping.items():
        if value == n_inputs: 
            latent_upper_lim = key
            break
    limits = set_limits([0, 1, 0, latent_upper_lim])

    # Create initial population of parents
    parents = initialize_candidates(cand_size, pop_size, limits)
    
    for g in tqdm(range(gens)):
        # print('Generation ', g+1, ':\n')
        trials = []
        F = get_F()
        for index, candidate in enumerate(parents):
            while(True):
                # Perform mutation and crossover to get trial and check constraints on trial
                mutant = mutation(parents, index, K, F, candidate)
                trial = crossover(mutant, candidate)
                if (check_constraints(trial, limits) == True): break
                retry_count += 1
            trials.append(np.array(trial))
        # print("Trial vectors generated\n")
        # NSDE selection
        next_gen = nsde(parents+trials, functions)
        parents = next_gen[:]
    print(retry_count)
    return next_gen
