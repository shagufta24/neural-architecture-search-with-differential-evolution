from moode_funcs_2 import *
from test_funcs import *
import numpy as np
import time

def plot_2_obj(population, functions):
    x_vals = [functions[0](candidate) for candidate in population]
    y_vals = [functions[1](candidate) for candidate in population]
    plt.scatter(x_vals, y_vals)
    plt.xlabel("Function 1")
    plt.ylabel("Function 2")
    plt.title("Pareto front")
    plt.show()

if __name__ == "__main__":
    population_size = 50
    candidate_size = 9
    no_of_inputs = 10
    num_of_gens = 100
    func = Chanking_Haimes()
    func_name = 'Chanking Haimes'

    start_time = time.time()
    next_gen = moode(population_size, candidate_size, no_of_inputs, num_of_gens, [func.f1,func.f2])
    print("--- %s seconds ---" % (time.time() - start_time))

    # Plotting
    plot_2_obj(next_gen, [func.f1,func.f2])