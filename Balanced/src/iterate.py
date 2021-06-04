# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:59:35 2021

@author: ryneo
"""

import gurobipy as gp
from gurobipy import GRB
import math

def iterate_U_and_L(m, population, U, L, k, districts):
    DG = m._DG
    
    dist_pop = []
    
    for j in range(k):
        dist_pop.append(0)
        
    for j in range(k):
        for i in DG.nodes:
            if i in districts[j]:
                dist_pop[j] += population[i]
            
    max_pop = max(dist_pop)
    min_pop = min(dist_pop)
    
    ideal = math.floor(sum(population) / k)
    
    if (max_pop - ideal) >= (ideal - min_pop):
        dev = max_pop - ideal
    
    else:
        dev = ideal - min_pop
        
    m._UB.ub =  ideal + (dev - 1)
    m._LB.lb = ideal - (dev - 1)
    print(m._UB.ub)
    print(m._LB.lb)
    
    m.update()