import numpy as np

class Binh_Korn(object):
    limits = [0, 5, 0, 3]
    
    def f1(self, vector):
        x = vector[0]
        y = vector[1]
        return (4*np.square(x) + 4*np.square(y))

    def f1_constraints(self, x, y):
        if (np.square((x-5)) + np.square(y) <= 25):
            return True
        else: return False

    def f2(self, vector):
        x = vector[0]
        y = vector[1]
        return (np.square((x-5)) + np.square((y-5)))

    def f2_constraints(self, x, y):
        if (np.square((x-8)) + np.square((y+3)) >= 7.7):
            return True
        else: return False

class Chanking_Haimes(object):
    limits = [-20, 20, -20, 20]
    
    def f1(self, vector):
        x = vector[0]
        y = vector[1]
        return (2 + np.square((x-2)) + np.square((y-1)))

    def f1_constraints(self, x, y):
        if (np.square(x) + np.square(y) <= 225):
            return True
        else: return False

    def f2(self, vector):
        x = vector[0]
        y = vector[1]
        return (9*x - np.square((y-1)))

    def f2_constraints(self, x, y):
        if (x - 3*y + 10 <= 0):
            return True
        else: return False

class Constr_Ex(object):
    limits = [0.1, 1, 0, 5]
    
    def f1(self, vector):
        x = vector[0]
        y = vector[1]
        return (x)

    def f1_constraints(self, x, y):
        if (y + 9*x >= 6):
            return True
        else: return False

    def f2(self, vector):
        x = vector[0]
        y = vector[1]
        return ((1+y)/x)

    def f2_constraints(self, x, y):
        if (-y + 9*x >= 1):
            return True
        else: return False