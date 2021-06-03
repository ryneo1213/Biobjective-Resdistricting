# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:00:01 2021

@author: ryneo
"""

import networkx as nx

# how many people are reachable from v in G[S]? Uses BFS
def reachable_population(G, population, S, v):
    pr = 0 # population reached
    if not S[v]:
        return 0
    
    visited = [False for i in G.nodes]
    child = [v]
    visited[v] = True
    while child:
        parent = child
        child = list()
        for i in parent:
            pr += population[i]
            for j in G.neighbors(i):
                if S[j] and not visited[j]:
                    child.append(j)
                    visited[j] = True
    return pr   

def do_Labeling_DFixing(m, G, ordering, k):
    DFixings = 0
    m.update()
    for p in range(G.number_of_nodes()):
        i = ordering[p]
        for j in range(p+1,k):
            if m._X[i,j].UB > 0.5:
                m._X[i,j].UB = 0
                DFixings += 1
    m.update()
    return DFixings


def do_Labeling_ZFixing(m, G, k):
    ZFixings = 0
    for u,v in G.edges:
        for j in range(k):
            if m._X[u,j].UB < 0.5 or m._X[v,j].LB > 0.5:
                m._Z[u,v,j].UB = 0
                ZFixings += 1
            elif m._X[u,j].LB > 0.5 and m._X[v,j].UB < 0.5:
                m._Z[u,v,j].LB = 1
                ZFixings += 1
    m.update()
    return ZFixings


def do_Labeling_LFixing(m, G, population, L, ordering, k):
    LFixings = 0
    
    # find "back" of ordering B = {v_q, v_{q+1}, ..., v_{n-1} }
    n = G.number_of_nodes()
    S = [False for v in G.nodes]
    for p in range(n):
        v_pos = n - p - 1
        v = ordering[v_pos]
        S[v] = True
        pr = reachable_population(G, population, S, v)
        if pr >= L:
            q = v_pos + 1
            break
    
    # none of the vertices at back (in B) can root a district. 
    for p in range(q,n):
        i = ordering[p]
        for j in range(k):
            if m._R[i,j].UB > 0.5:
                m._R[i,j].UB = 0
                LFixings += 1
    
    # vertex v_{q-1} cannot root districts {0, 1, ..., k-2}
    # vertex v_{q-2} cannot root districts {0, 1, ..., k-3}
    # ... 
    # vertex v_{q-t} cannot root districts {0, 1, ..., k-t-1}
    # ...
    # vertex v_{q-(k-1)} cannot root district {0}
    for t in range(1,k):
        i = ordering[q-t]
        for j in range(k-t):
            if m._R[i,j].UB > 0.5:
                m._R[i,j].UB = 0
                LFixings += 1
    
    m.update()
    return LFixings 

def do_Labeling_UFixing(m, DG, population, U, ordering, k):
    UFixings_X = 0
    UFixings_R = 0
    DG = m._DG
    for (i,j) in DG.edges:
        DG[i][j]['ufixweight'] = population[j] # weight of edge (i,j) is population of its head j
    
    for j in range(k):
        
        v = ordering[j]
        dist = nx.shortest_path_length(DG,source=v,weight='ufixweight')
        
        if j == 0:
            min_dist = U+1
        else:
            min_dist = min( dist[ordering[t]] + population[v] for t in range(j) )
        
        if min_dist <= U:
            break
        
        if m._R[v,j].LB < 0.5:
            m._R[v,j].LB = 1
            UFixings_R += 1
            
        if m._X[v,j].LB < 0.5:
            m._X[v,j].LB = 1
            UFixings_X += 1
            
        for t in range(k):
            if t != j and m._X[v,t].UB > 0.5:
                m._X[v,t].UB = 0
                UFixings_X += 1
        
        for i in DG.nodes:
            if i != v and m._R[i,j].UB > 0.5:
                m._R[i,j].UB = 0
                UFixings_R += 1
        
        for i in DG.nodes:
            if i != v and dist[i] + population[v] > U and m._X[i,j].UB > 0.5:
                m._X[i,j].UB = 0
                UFixings_X += 1
        
    m.update()
    return UFixings_X, UFixings_R    