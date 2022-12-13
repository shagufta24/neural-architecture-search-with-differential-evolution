import sys
import numpy as np
from random import SystemRandom
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

K = 0.5
crossover_probability = 0.75
obj_func_1 = []
obj_func_2 = []

# Upper and lower bounds for variables
class set_limits(object):
    def __init__(self, lims):
        self.io_min = lims[0]
        self.io_max = lims[1]
        self.latent_min = lims[2]
        self.latent_max = lims[3]

# Initializes the fixed mapping to map floats to integers
# Divides the range (0-1) into fixed number of bins
def initialize_mapping(n_inputs):
    # global rand_nums
    # rand_nums = np.array([SystemRandom().uniform(0, 1) for i in range(2*n_inputs)])
    # rand_nums.sort()
    global float_nums
    # Precision of each value is 2
    float_nums = np.array([float("{:.2f}".format(i*(1/(2*n_inputs+1)))) for i in range(2*n_inputs+1)])
    global mapping
    mapping = {float_nums[i]: i for i in range(2*n_inputs+1)}

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

# Picks 3 random candidates from parents pool and performs mutation
def mutation(parents, index, K, F, candidate):
    parents_minus_i = parents[:]
    parents_minus_i.pop(index)
    r1,r2,r3 = SystemRandom().sample(parents_minus_i, 3)
    mutant = candidate + K*(r1 - candidate) + F*(r2 - r3)
    return mutant

# Return true if each value (io layer nodes or latent nodes) in the candidate is within the bounds
def check_constraints(candidate, limits):
    # print(candidate)
    for x in candidate[:-1]:
        if not(limits.io_min <= x <= limits.io_max):
            return False
    if not(limits.latent_min <= candidate[-1] <= limits.latent_max):
        return False
    return True

# Perform crossover between 2 vectors and generate trial vector
def crossover(mutant, candidate):
    rand = SystemRandom().random()
    trial = [mutant[i] if rand <= crossover_probability else candidate[i] for i in range(len(candidate))]
    return trial

# Mapping of floating values in candidate to integer values using fixed mapping
def float_to_int_mapping(vector):
    mapped = []
    for x in vector:
        closest_id = (np.abs(float_nums - x)).argmin()
        mapped.append(float_nums[closest_id])
    int_vector = []
    for i in mapped:
        int_vector.append(mapping[i])
    return int_vector

# Checking if one candidate pareto domainates another based on obj function values
def dominate(cand1_index, cand2_index, obj_func_vals):
    flag = 0
    for f in functions:
        v1_fitness = obj_func_vals[0][cand1_index]
        v2_fitness = obj_func_vals[0][cand2_index]
        if v1_fitness > v2_fitness: return False
        if v1_fitness < v2_fitness: flag = 1
    if flag == 1: return True
    return False

def get_front(population, mapped_pop, obj_func_vals):
    # Initialize n and S
    pop_size = len(population)
    n = [0] * pop_size
    S = []

    # Pareto dominance to choose front members
    # Computing ni and Si for each member
    for index1 in range(pop_size):
        Si = []
        for index2 in range(pop_size):
            if (dominate(index1, index2, obj_func_vals)):
                Si.append(index1)
            elif dominate(index2, index1, obj_func_vals): 
                n[index1] += 1
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

def crowding_distance(population, mapped_pop, obj_func_vals):
    # Array of Di for each member i of front k
    crowd_dist = [sys.maxsize] * len(population)

    # For each function
    for func_values in obj_func_vals:
        # Sort population in ascending order of objective function values
        # func_values = []
        # for member in mapped_pop:
        #     func_values.append(f(member))
        
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

# To find the next generation of candidates from the parents+trials pool
def nsde(population, functions, gen):
    print("NSDE Algo")
    pop_size = len(population)/2 # Size of new generation
    next_gen = []

    # Map all candidates from float to integer space
    mapped_pop = []
    for vector in population:
        mapped_pop.append(float_to_int_mapping(vector))
    # print('Population mapped to integer space\n')

    # Compute and store obj function values for all candidates
    if (gen == 0):
        obj_func_1 = [functions[0](candidate) for candidate in mapped_pop] # training loss
        obj_func_2 = [functions[1](candidate) for candidate in mapped_pop] # no of latent nodes
    else:
        # We only recalculate the trial vector function values, since parent vector function values
        # were computed in previous generation
        for i in range(pop_size/2-1, pop_size):
            obj_func_1[i] = functions[0](mapped_pop[i])
            obj_func_2[i] = functions[1](mapped_pop[i])
            
    unfilled_spots = pop_size # initially all spots are empty
    while(len(next_gen) < pop_size):
        # Generate a front
        front, mapped_front, new_population, new_mapped_pop = get_front(population, mapped_pop, [obj_func_1, obj_func_2])

        # If size of front is smaller than the remaining spots, add all front members to next gen
        if (len(front) <= unfilled_spots):
            for member in front:
                next_gen.append(member)
            unfilled_spots -= len(front)
            # print('Added all members of front to next gen\n')

        # If size of front is greater than the remaining spots, choose members using crowding distance algo
        else:
            # print('Crowding distance algo\n')
            result = crowding_distance(front, mapped_front, [obj_func_1, obj_func_2])[:int(unfilled_spots)]
            for member in result:
                next_gen.append(member)
            unfilled_spots -= len(result)

        # print('NSDE done\n')
        population = new_population[:]
        mapped_pop = new_mapped_pop[:]
    return next_gen

# Replace every negative value with a random value between 0 and 1
def approx_trial(trial):
    return [SystemRandom().uniform(0, 1) if val < 0 else val for val in trial]

def moode(pop_size, cand_size, n_inputs, gens, functions):
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
    approx_trial_vectors = 0
    for g in range(gens):
        start_time = time.time()
        print('Generation ', g+1, ':\n')
        trials = []
        F = get_F()
        for index, candidate in enumerate(parents):
            retry_count = 0
            while(True):
                # Perform mutation and crossover to get trial and check constraints on trial
                mutant = mutation(parents, index, K, F, candidate)
                trial = crossover(mutant, candidate)
                if (check_constraints(trial, limits) == True): break
                retry_count += 1
                # If too many retries, replace negative values with random values in trial vector
                # With threshold = 1000, around 8-10% vectors are approximated
                # With threshold = 1500, around 8-10% vectors are approximated
                if (retry_count == 1500):
                    trial = approx_trial(trial)
                    approx_trial_vectors += 1
                    break
            trials.append(np.array(trial))
        # print("Trial vectors generated\n")
        # print("Retries: ", retry_count)
        print("--- %s seconds ---" % (time.time() - start_time))
        # NSDE selection
        # population = parents + trials
        # next_gen = nsde(population, functions, g)
        # parents = next_gen[:]
    # return next_gen
    print("Total approx:", approx_trial_vectors)
    return 0
