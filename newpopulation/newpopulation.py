# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:32:30 2021

@author: ryneo
"""

import gurobipy as gp
from gurobipy import GRB
import math

def add_pop_constraints(m, population, deviation, k, G, max_dev):
            dist_pop = []
            DG = m._DG
            
            for j in range(k):
                dist_pop.append(0)
            
            for j in range(k):
                for i in range(G.number_of_nodes()):
                        dist_pop[j] += population[i] * m._X[i,j]
                        
            m._P = m.addVars(range(k), range(k), vtype = GRB.INTEGER)
            
            for j in range(k):
                for i in range(k):
                    m._P[i,j] = dist_pop[j] - dist_pop[i]
             
            m.addConstrs(m._P[i,j] <= max_dev for i in range(k) for j in range(k))
            
            m.update()