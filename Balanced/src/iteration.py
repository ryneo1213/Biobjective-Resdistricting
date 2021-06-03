# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:59:35 2021

@author: ryneo
"""

import gurobipy as gp
from gurobipy import GRB
import math

def iterate_U_and_L(m, population, U, L, k):
    
    max_pop = max(m._P)
    print(max_pop)
    min_pop = min(m._P)
    print(min_pop)
    
    ideal = math.floor(sum(population) / k)
    
    if (max_pop - ideal) >= (ideal - min_pop):
        dev = max_pop - ideal
    
    else:
        dev = ideal - max_pop
        
    U = ideal + (dev - 1)
    L = ideal - (dev + 1)
    m._U.ub = U
    m._L.lb = L
    
    m.update()