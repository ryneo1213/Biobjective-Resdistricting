# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:55:41 2021

@author: ryneo
"""
import gurobipy as gp
from gurobipy import GRB


def most_possible_nodes_in_one_district(population, U):
    cumulative_population = 0
    num_nodes = 0
    for ipopulation in sorted(population):
        cumulative_population += ipopulation
        num_nodes += 1
        if cumulative_population > U:
            return num_nodes - 1

def find_ai(DG, i, k, population, U, sol_a):
    
    #Create Model m with DG
    m = gp.Model()
    
    #Create Variables x and f
    x = m.addVars(DG.nodes, vtype = GRB.BINARY)
    f = m.addVars(DG.edges, vtype=GRB.CONTINUOUS)
    ideal = sum(population) / k
    
    M = most_possible_nodes_in_one_district(population, U) - 1
    
    m.setObjective((gp.quicksum(population[j] * x[j] for j in DG.nodes) - ideal), GRB.MINIMIZE)
    
    
    m.addConstr(gp.quicksum(population[j] * x[j] for j in DG.nodes) >= ideal)
    m.addConstr(x[i] == 1)
    
    #Consume one unit of flow at selected nodes
    m.addConstrs(gp.quicksum((f[v,j] - f[j,v]) for v in DG.neighbors(j)) == x[j] for j in DG.nodes if j != i)
    
    #If vertex is not selected, then no flow can enter it
    m.addConstrs(gp.quicksum(f[v,j] for v in DG.neighbors(j)) <= M * x[j] for j in DG.nodes if j !=i)
    
    
    m.optimize()
    
    solution = []
    for j in DG.nodes:
        if x[j].X > 0.5:
            solution.append(j)
    
            
    sol_a.append(solution)
    
    return m.objVal

def find_bi(DG, i, k, population, U, sol_b):
    
    #Create Model m with DG
    m = gp.Model()
    
    #Create Variables x and f
    x = m.addVars(DG.nodes, vtype = GRB.BINARY)
    f = m.addVars(DG.edges, vtype=GRB.CONTINUOUS)
    ideal = sum(population) / k
    
    M = most_possible_nodes_in_one_district(population, U) - 1
    
    m.setObjective(ideal - (gp.quicksum(population[j] * x[j] for j in DG.nodes)), GRB.MINIMIZE)
    
    
    m.addConstr(gp.quicksum(population[j] * x[j] for j in DG.nodes) <= ideal)
    m.addConstr(x[i] == 1)
    
    #Consume one unit of flow at selected nodes
    m.addConstrs(gp.quicksum((f[v,j] - f[j,v]) for v in DG.neighbors(j)) == x[j] for j in DG.nodes if j != i)
    
    #If vertex is not selected, then no flow can enter it
    m.addConstrs(gp.quicksum(f[v,j] for v in DG.neighbors(j)) <= M * x[j] for j in DG.nodes if j !=i)
    
    
    m.optimize()
    
    solution = []
    for j in DG.nodes:
        if x[j].X > 0.5:
            solution.append(j)
            
    sol_b.append(solution)
    
    return m.objVal
