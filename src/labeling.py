import gurobipy as gp
from gurobipy import GRB 
 
def add_objective(m, G, k):
    # Y[i,j] = 1 if edge {i,j} is cut
    m._Y = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstrs( m._X[i,v]-m._X[j,v] <= m._Y[i,j] for i,j in G.edges for v in range(k))
    m.setObjective( gp.quicksum(m._Y), GRB.MINIMIZE )

    
def add_extended_objective(m, G, k):
    # Z[i,j,v] = 1 if edge (i,j) is cut because i->v but j!->v
    m._Z = m.addVars(G.edges, range(k), vtype=GRB.BINARY)
    m.addConstrs( m._X[i,v]-m._X[j,v] <= m._Z[i,j,v] for i,j in G.edges for v in range(k))
    m.setObjective( gp.quicksum(m._Z), GRB.MINIMIZE)
            
def most_possible_nodes_in_one_district(population, U):
    cumulative_population = 0
    num_nodes = 0
    for ipopulation in sorted(population):
        cumulative_population += ipopulation
        num_nodes += 1
        if cumulative_population > U:
            return num_nodes - 1
             
def add_scf_constraints(m, G, U):
    DG = m._DG
    k = m._k
    
    # f[u,v] = amount of flow sent across arc uv
    f = m.addVars(DG.edges, vtype=GRB.CONTINUOUS)
    
    # compute big-M    
    M = most_possible_nodes_in_one_district(m._population, U) - 1
    
    # the following constraints are weaker than some in the orbitope EF
    m.addConstrs( gp.quicksum(m._R[i,j] for i in DG.nodes)==1 for j in range(k) )
    m.addConstrs( m._R[i,j] <= m._X[i,j] for i in DG.nodes for j in range(k) )  
    
    # if not a root, consume some flow.
    # if a root, only send out so much flow.
    m.addConstrs( gp.quicksum(f[u,v]-f[v,u] for u in DG.neighbors(v)) >= 1 - M * gp.quicksum(m._R[v,j] for j in range(k)) for v in G.nodes)
    
    # do not send flow across cut edges
    m.addConstrs( f[i,j] + f[j,i] <= M*(1 - gp.quicksum( m._Z[i,j,v] for v in range(k) )) for (i,j) in G.edges)