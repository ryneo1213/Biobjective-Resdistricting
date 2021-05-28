# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:32:30 2021

@author: ryneo
"""

import gurobipy as gp
from gurobipy import GRB

def add_pop_constraints(m, population, deviation, k):
    