from moode_funcs import *
from test_funcs import *
import numpy as np
import time

if __name__ == "__main__":
    population_size = 100
    num_of_gens = 100
    func = Chanking_Haimes()
    func_name = 'Chanking Haimes'

    start_time = time.time()
    moode(population_size, num_of_gens, func_name, [func.f1,func.f2], func.limits, [func.f1_constraints, func.f2_constraints])
    print("--- %s seconds ---" % (time.time() - start_time))
