# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:56:29 2021

@author: ryneo
"""

###########################
# Imports
###########################  

import gurobipy as gp
from gurobipy import GRB 
import matplotlib.pyplot as plt
import numpy as np

from datetime import date
import math
import networkx as nx
import csv
import time
import json
import sys
import os

import labeling
import ordering
import fixing
import mindev

from gerrychain import Graph
import geopandas as gpd


################################################
# Summarize computational results to csv file
################################################ 

from csv import DictWriter
def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)
        
        
################################################
# Writes districting solution to json file
################################################ 

def export_to_json(G, districts, filename):
    with open(filename, 'w') as outfile:
        soln = {}
        soln['nodes'] = []
        for j in range(len(districts)):
            for i in districts[j]:
                soln['nodes'].append({
                        'name': G.nodes[i]["NAME10"],
                        'index': i,
                        'district': j
                        })
        json.dump(soln, outfile)
               

################################################
# Draws districts and saves to png file
################################################ 

def export_to_png(G, df, districts, filename):
    
    assignment = [ -1 for u in G.nodes ]
    
    for j in range(len(districts)):
        for i in districts[j]:
            geoID = G.nodes[i]["GEOID10"]
            for u in G.nodes:
                if geoID == df['GEOID10'][u]:
                    assignment[u] = j
    
    if min(assignment[v] for v in G.nodes) < 0:
        print("Error: did not assign all nodes in district map png.")
    else:
        df['assignment'] = assignment
        my_fig = df.plot(column='assignment').get_figure()
        RESIZE_FACTOR = 3
        my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
        plt.axis('off')
        my_fig.savefig(filename)


################################################
# Draws max B set and saves to png file
################################################ 

def export_B_to_png(G, df, B, filename):
    
    B_geoids = [ G.nodes[i]["GEOID10"] for i in B ]
    df['B'] = [1 if df['GEOID10'][u] in B_geoids else 0 for u in G.nodes]
        
    my_fig = df.plot(column='B').get_figure()
    RESIZE_FACTOR = 3
    my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
    plt.axis('off')
    my_fig.savefig(filename)
    
def export_mindev_to_png(G, df, sol, filename):
    
    sol_geoids = [ G.nodes[i]["GEOID10"] for i in sol]
    df['sol'] = [1 if df['GEOID10'][u] in sol_geoids else 0 for u in G.nodes]
        
    my_fig = df.plot(column='sol').get_figure()
    RESIZE_FACTOR = 3
    my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
    plt.axis('off')
    my_fig.savefig(filename)
    

###########################
# Hard-coded inputs
###########################  

state_codes = {
    'WA': '53', 'DE': '10', 'WI': '55', 'WV': '54', 'HI': '15',
    'FL': '12', 'WY': '56', 'NJ': '34', 'NM': '35', 'TX': '48',
    'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
    'PA': '42', 'AK': '02', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '08',
    'CA': '06', 'AL': '01', 'AR': '05', 'VT': '50', 'IL': '17', 'GA': '13',
    'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '04', 'ID': '16', 'CT': '09',
    'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
    'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
    'SC': '45', 'KY': '21', 'OR': '41', 'SD': '46'
}

number_of_congressional_districts = {
    'WA': 10, 'DE': 1, 'WI': 8, 'WV': 3, 'HI': 2,
    'FL': 27, 'WY': 1, 'NJ': 12, 'NM': 3, 'TX': 36,
    'LA': 6, 'NC': 13, 'ND': 1, 'NE': 3, 'TN': 9, 'NY': 27,
    'PA': 18, 'AK': 1, 'NV': 4, 'NH': 2, 'VA': 11, 'CO': 7,
    'CA': 53, 'AL': 7, 'AR': 4, 'VT': 1, 'IL': 18, 'GA': 14,
    'IN': 9, 'IA': 4, 'MA': 9, 'AZ': 9, 'ID': 2, 'CT': 5,
    'ME': 2, 'MD': 8, 'OK': 5, 'OH': 16, 'UT': 4, 'MO': 8,
    'MN': 8, 'MI': 14, 'RI': 2, 'KS': 4, 'MT': 1, 'MS': 4,
    'SC': 7, 'KY': 6, 'OR': 5, 'SD': 1
}

default_config = {
    'state' : 'OK',
    'level' : 'county',
}

available_config = {
    'state' : { key for key in state_codes.keys() },
    'level' : {'county', 'tract'},
}


###############################################
# Read configs/inputs and set parameters
############################################### 

# read configs file and load into a Python dictionary

if len(sys.argv)>1:
    # name your own config file in command line, like this: 
    #       python main.py usethisconfig.json
    # to keep logs of the experiments, redirect to file, like this:
    #       python main.py usethisconfig.json 1>>log_file.txt 2>>error_file.txt
    config_filename = sys.argv[1] 
else:
    config_filename = 'config.json' # default
    
print("Reading config from",config_filename)    
config_filename_wo_extension = config_filename.rsplit('.',1)[0]
configs_file = open(config_filename,'r')
batch_configs = json.load(configs_file)
configs_file.close()

# create directory for results
path = os.path.join("..", "results_for_" + config_filename_wo_extension) 
os.mkdir(path) 

# print results to csv file
today = date.today()
today_string = today.strftime("%Y_%b_%d") # Year_Month_Day, like 2019_Sept_16
results_filename = "../results_for_" + config_filename_wo_extension + "/results_" + config_filename_wo_extension + "_" + today_string + ".csv" 

# prepare csv file by writing column headers
with open(results_filename,'w',newline='') as csvfile:   
    my_fieldnames = ['run','iteration','state','level'] # configs
    my_fieldnames += ['k', 'max_dev', 'n','m'] # params
    my_fieldnames += ['heur_obj', 'heur_time', 'heur_iter'] # heuristic info
    my_fieldnames += ['B_q', 'B_size', 'B_time', 'B_timelimit'] # max B info
    my_fieldnames += ['DFixings', 'LFixings', 'UFixings_X', 'UFixings_R', 'ZFixings'] # fixing info
    my_fieldnames += ['MIP_obj','MIP_bound','MIP_time', 'MIP_timelimit', 'MIP_status', 'MIP_nodes', 'connected'] # MIP info
    my_fieldnames += ['max_dist', 'min_dist', 'found_dev']
    writer = csv.DictWriter(csvfile, fieldnames = my_fieldnames)
    writer.writeheader()
    
############################################################
# Run experiments for each config in batch_config file
############################################################

for key in batch_configs.keys(): 
      
    # get config and check for errors
    config = batch_configs[key]
    print("In run",key,"using config:",config,end='.')
    for ckey in config.keys():
        if config[ckey] not in available_config[ckey]:
            errormessage = "Error: the config option"+ckey+":"+config[ckey]+"is not known."
            sys.exit(errormessage)
    print("")
    
    # fill-in unspecified configs using default values
    for ckey in available_config.keys():
        if ckey not in config.keys():
            print("Using default value",ckey,"=",default_config[ckey],"since no option was selected.")
            config[ckey] = default_config[ckey]
        
    # initialize dictionary to store this run's results
    result = config
    result['run'] = key            
                   
    # read input data
    state = config['state']
    code = state_codes[state]
    level = config['level']
    G = Graph.from_json("../data/"+level+"/dual_graphs/"+level+code+".json")
    DG = nx.DiGraph(G) # bidirected version of G
    df = gpd.read_file("../data/"+level+"/shape_files/"+state+"_"+level+".shp")      

    # set parameters
    k = number_of_congressional_districts[state]        
    population = [G.nodes[i]['TOTPOP'] for i in G.nodes()]    
    deviation = 0.01
    U = math.floor((1 + deviation) * (sum(population) / k))
    L = math.ceil((1 - deviation) * (sum(population) / k))    
    ideal = math.floor(sum(population) / k)
    max_dev = math.floor(deviation * ideal)

    print("k =",k)
    result['k'] = k
    result['n'] = G.number_of_nodes()
    result['m'] = G.number_of_edges()
    
    #mindev problem
    
    a = []
    b = []
    sol_a = []
    sol_b = []
    sol = []
    
    #append each optimal district for each node both from above and below
    for i in range(G.number_of_nodes()):
        a.append(mindev.find_ai(DG, i, k, population, U, sol_a))
        b.append(mindev.find_bi(DG, i, k, population, U, sol_b))
    
    d = 0
    
    for i in G.nodes:
        if a[i] > d and b[i] > d:
            if a[i] <= b[i]:
                d = a[i]
                sol = sol_a[i]
            else:
                d = b[i]
                sol = sol_b[i]
                
    fn_d = "../" + "results_for_" + config_filename_wo_extension + "/" + result['state'] + "-" + result['level'] + "-d.png"  

    export_mindev_to_png(G, df, sol, fn_d)
    minimum_deviation = d * k / (k-1)
    
    # read heuristic solution from external file (?)

    heuristic_file = open('../data/'+level+"/heuristic/heur_"+state+"_"+level+".json",'r')
    heuristic_dict = json.load(heuristic_file)       
    heuristic_districts = [ [node['index'] for node in heuristic_dict['nodes'] if node['district']==j ] for j in range(k) ]
    result['heur_obj'] = heuristic_dict['obj']
    result['heur_time'] = heuristic_dict['time']
    result['heur_iter'] = heuristic_dict['iterations']


    ############################ ONLY DO ONCE
    # Build base model
    ############################   
    
    m = gp.Model()
    m._DG = DG
   
    # X[i,j]=1 if vertex i is assigned to district j in {0,1,2,...,k-1}
    m._X = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY)
    m._R = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY)
    
    # Each vertex i assigned to one district
    m.addConstrs(gp.quicksum(m._X[i,j] for j in range(k)) == 1 for i in DG.nodes)
    
    #Add P Constraints
    m._P = m.addVars(range(k), vtype = GRB.INTEGER)
    m.addConstrs(gp.quicksum(population[i] * m._X[i,j] for i in DG.nodes) == m._P[j] for j in range(k))
    
    # Max and minimum population variables
    m._max_pop = m.addVar(vtype=GRB.INTEGER)
    m._min_pop = m.addVar(vtype=GRB.INTEGER)
    m.update()
    
    #Population constraints
    m.addConstrs(m._P[j] <= m._max_pop for j in range(k))
    m.addConstrs(m._P[j] >= m._min_pop for j in range(k))
    dev_constr = m.addConstr(m._max_pop - m._min_pop <= max_dev)
    
    
    # Set tolerance requirements
    m.Params.IntFeasTol = 1.e-9
    m.Params.FeasibilityTol = 1.e-9
    m.update()    
                
    ############################################   ONLY DO ONCE   
    # Add (extended?) objective 
    ############################################         
    

    labeling.add_extended_objective(m, G, k)
            
    ############################################   ONCE
    # Vertex ordering and max B problem 
    ############################################  
    
    (B, result['B_q'], result['B_time'], result['B_timelimit']) = ordering.solve_maxB_problem(DG, population, m, k, L, heuristic_districts)
    
    
    # draw set B on map and save
    fn_B = "../" + "results_for_" + config_filename_wo_extension + "/" + result['state'] + "-" + result['level'] + "-maxB.png"       
    export_B_to_png(G, df, B, fn_B)
    
    result['B_size'] = len(B)
    
    vertex_ordering = ordering.find_ordering(B, DG, population)
    position = ordering.construct_position(vertex_ordering)
    
    print("Vertex ordering =", vertex_ordering)  
    print("Position vector =", position)
    print("Set B =", B)
    
    ####################################   
    # Contiguity constraints
    ####################################      
            
    m._callback = None
    m._population = population
    m._k = k
        
                        
    
    labeling.add_scf_constraints(m, G, U)
    
    ######START LOOP########
    
    statusCode = 2
    iteration = 1
    Xs = []
    Ys = []
    points = 0
    currY = 0
    while (statusCode == 2):        
    
        ####################################   
        # Variable fixing
        ####################################    
        
        result['DFixings'] = fixing.do_Labeling_DFixing(m, G, vertex_ordering, k)
        result['LFixings'] = fixing.do_Labeling_LFixing(m, G, population, vertex_ordering, k, L)
        (result['UFixings_X'], result['UFixings_R']) = fixing.do_Labeling_UFixing(m, DG, population, vertex_ordering, k, U)
        result['ZFixings'] = fixing.do_Labeling_ZFixing(m, G, k)
    
            
        
        ####################################   
        # Inject heuristic warm start
        ####################################    
        
                        
        center_positions = [ min( position[v] for v in heuristic_districts[j] ) for j in range(k) ] 
        cplabel = { center_positions[j] : j for j in range(k) }
    
        # what node r will root the new district j? The one with earliest position.
        for j in range(k):
            min_cp = min(center_positions)
            r = vertex_ordering[min_cp]
            old_j = cplabel[min_cp]
            
            for i in heuristic_districts[old_j]:
                m._X[i,j].start = 1
                
            center_positions.remove(min_cp)
                
        
        ####################################   
        # Solve MIP
        ####################################  
        
        result['MIP_timelimit'] = 3600 # set a one hour time limit
        m.Params.TimeLimit = result['MIP_timelimit']
        m.Params.Method = 3 # use concurrent method for root LP. Useful for degenerate models
        
        start = time.time()
        m.optimize(m._callback)
        end = time.time()
        result['MIP_time'] = '{0:.2f}'.format(end-start)
        
        
        statusCode = int(m.status)
        result['MIP_status'] = statusCode
        result['MIP_nodes'] = int(m.NodeCount)
        result['MIP_bound'] = m.objBound
        result['max_dev'] = max_dev
        
        # report best solution found
        if m.SolCount > 0:
            result['MIP_obj'] = int(m.objVal)
            Ys.append(int(m.objVal)) 
            points += 1
    
    
            labels = [ j for j in range(k) ]
                
            districts = [ [ i for i in DG.nodes if m._X[i,j].x > 0.5 ] for j in labels]
            print("best solution (found) =",districts)
            
            fn = "../" + "results_for_" + config_filename_wo_extension + "/" + result['state'] + "-" + result['level']
            
            # export solution to .json file
            json_fn = fn + "_" + str(iteration) + ".json"
            export_to_json(G, districts, json_fn)
            
            # export solution to .png file (districting map)
            png_fn = fn + "_" + str(iteration) + ".png"
            export_to_png(G, df, districts, png_fn)
            
            # is solution connected?
            connected = True
            for district in districts:
                if not nx.is_connected(G.subgraph(district)):
                    connected = False
            result['connected'] = connected
            
            most = m._P[0].X
            least = m._P[0].X
            for j in range(k):
                
                if m._P[j].X > most:
                    most = m._P[j].X
                
                if m._P[j].X < least:
                    least = m._P[j].X
                
            
            result['iteration'] = iteration
            result['max_dist'] = most
            result['min_dist'] = least
            result['found_dev'] = most - least
            Xs.append(int(most-least))
            
        else:
            result['MIP_obj'] = 'no_solution_found'
            result['connected'] = 'n/a'
            
    
            
        ####################################   
        # Summarize results of this run to csv file
        ####################################
        
        append_dict_as_row(results_filename,result,my_fieldnames)
        
        max_dev = most - least
        dev_constr.setAttr(GRB.Attr.RHS, max_dev - 1)
        
        m.update()
        
        iteration += 1
        
    openX = []
    openY = []
    temp = points
    
    i = points - 1
    while i >= 0:
        if Ys[i] == currY:
            Xs.pop(i)
            Ys.pop(i)
            points -= 1
        else:
            currY = Ys[i]
        i -= 1
    
    for i in range(points - 1):
        openX.append(Xs[i])
        openY.append(Ys[i + 1])
        
        
    print(Xs)
    print(Ys)
    print(openX)
    
    #Plotting Pareto Chart
    fig, ax = plt.subplots(figsize =(12,6))
    ax.set_xlim(-750, ideal * 0.011)
    ax.set_ylim(min(Ys) - 2, max(Ys) + 5)
    ax.scatter(openX, openY, facecolors = 'w', edgecolors = 'b')
    ax.scatter(Xs, Ys)
    ax.scatter(ideal * 0.01, Ys[0], edgecolors = 'b', facecolors = 'b')
    ax.title.set_text("Unbalanced Window Population Deviation vs. Cut Edges for " + state)
    
    for i in range(points - 1):
        plt.hlines(Ys[i + 1], Xs[i + 1], Xs[i], colors = 'blue')
        
    plt.hlines(Ys[0], max(Xs), ideal * 0.01, colors = 'blue')
    lastX = ideal * 0.01
    
    ax.scatter(lastX, min(Ys), facecolors = 'w', edgecolors = 'b')
   
    
    fig.canvas.draw()
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    inf = int(ylabels[-2])
    ylabels[-2] = r"$\infty$"
    ylabels.pop(-1)
    ax.set_yticklabels(ylabels)
    ax.get_yticklabels()[-2].set_fontsize(19)
    ax.xaxis.set_ticks(np.arange(0,8000,1000))
    ax.set(xlabel="Population Deviation",ylabel="Cut Edges")
    
    
    for i_x, i_y in zip(Xs, Ys):
        plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y), horizontalalignment = 'right')
        
    plt.hlines(inf, 0, minimum_deviation, colors = 'red', linestyles = 'dashed')
    ax.scatter(0, inf, facecolors = 'r', edgecolors = 'r')
    ax.scatter(minimum_deviation, inf, facecolors = 'w', edgecolors = 'r')
    
        
    a = minimum_deviation
    b = min(Xs)
    ybot = 1 - (5 / (max(Ys) - min(Ys) +7))
    ytop = 1- ((max(Ys) + 5 - inf) / (max(Ys) + 7 - min(Ys)))
    
    if statusCode == 9: 
        ax.axvspan(a, b, ymin = ybot, ymax = ytop, color='r', alpha=0.5, lw=0)
    
    
    fig.savefig(fn + "_pareto.png")