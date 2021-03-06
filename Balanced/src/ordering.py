import gurobipy as gp
from gurobipy import GRB 
import time

def sort_by_second(val):
    return val[1]


def construct_position(ordering):
    position = [-1 for i in range(len(ordering))]
    for p in range(len(ordering)):
        v = ordering[p]
        position[v] = p
    return position


def find_ordering(B, DG, population):
    V_B_with_population = [(i,population[i]) for i in DG.nodes if i not in B]
    V_B_with_population.sort(key=sort_by_second,reverse=True)
    return [v for (v,p) in V_B_with_population] + B

    

def solve_maxB_problem(DG, population, m, k, lowFix, heuristic_districts):
    m = gp.Model()
    m.params.LogToConsole = 0 # keep log to a minimum
    q = k
    
    # X[i,j]=1 if vertex i is assigned to bin j
    X = m.addVars(DG.nodes, range(q), vtype=GRB.BINARY)
    
    # B[i]=1 if vertex i is selected in set B
    B = m.addVars(DG.nodes, vtype=GRB.BINARY)
   
    # assignment constraints            
    m.addConstrs( gp.quicksum(X[i,j] for j in range(q)) == B[i] for i in DG.nodes )
                
    # bin population should be less than L
    m.addConstrs( gp.quicksum(population[i] * X[i,j] for i in DG.nodes) <= lowFix - 1 for j in range(q) )
    
    # bins shouldn't touch each other
    m.addConstrs( X[u,j] + B[v] <= 1 + X[v,j] for u,v in DG.edges for j in range(q) )
    
    # objective is to maximize size of set B
    m.setObjective( gp.quicksum( B ), GRB.MAXIMIZE )
    
    m.Params.MIPFocus = 1 # turn on MIPFocus
    B_timelimit = 60
    m.Params.timeLimit = B_timelimit # 60-second time limit
    
    # suggest a (partial) warm start
    if heuristic_districts is not None:
        for j in range(k):
            for i in heuristic_districts[j]:
                for t in range(q):
                    if t != j:
                        X[i,t].start = 0.0
    
    start = time.time()
    m.optimize()
    end = time.time()
    B_time = '{0:.2f}'.format(end-start)
    
    
    if m.status in { GRB.OPTIMAL, GRB.TIME_LIMIT }:
        B_sol = [i for i in DG.nodes if B[i].x > 0.5 ]
        print("max B obj val =",m.objVal)
    else:
        B_sol = list()
        
    return (B_sol, q, B_time, B_timelimit)  

