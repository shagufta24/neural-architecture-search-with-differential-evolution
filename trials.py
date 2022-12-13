
from random import SystemRandom
import numpy as np

# How to create mapping
rand_nums = [SystemRandom().uniform(0, 1) for i in range(75)]
rand_nums.sort()
mapping = {rand_nums[i]: i for i in range(75)}
mapping = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5:5}
rand_nums = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# How to map float to closest float then int
def float_to_int_mapping(vector):
    mapped = []
    for x in vector:
        closest_id = (np.abs(rand_nums - x)).argmin()
        mapped.append(rand_nums[closest_id])
    int_vector = []
    for i in mapped:
        int_vector.append(mapping[i])
    return int_vector
candidate = [0.173, 0.936, 0.453]
int_cand = float_to_int_mapping(candidate)

# How to sort a list based on another list
population = [[2, 3, 5], [6, 3, 9], [0, 8, 1]]
func_values = [4, 8, 2]
sorted_pop = [member for _, member in sorted(zip(func_values, population))]
# print(sorted_pop)

n_inputs = 7
float_nums = np.array([i*(1/(2*n_inputs+1)) for i in range(2*n_inputs+1)])
mapping = {float_nums[i]: i for i in range(2*n_inputs+1)}
# print(mapping)

a = [1, 2, 3, 4, 5]
b = ['a', 'b', 'c']
c = [6, 7, 8]
print(a+b+c)
        



