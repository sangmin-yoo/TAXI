import os
import time
import numpy as np
from concorde.tsp import TSPSolver

root_dir = os.getcwd()
tsp_dir = os.path.join(root_dir, "../tsplib")
concorde_dir = os.path.join(root_dir, "../pyconcorde/concorde")

def gen_tsp_data():
    ## Load a tsp library and Create a dictionary "tsp_database" of Name-X/Y coordinates.

    tsp_database = {}
    cities = os.listdir(tsp_dir)
    
    for city in cities:
        if city.endswith(".tsp"):
            with open(tsp_dir + '/' + city, 'r') as infile:
                lines = infile.readlines()
                for i in range(0, len(lines)):
                    line = lines[i]
                    if "DIMENSION" in line:
                        Dimension = line.strip().split(':')[1]
                        if not Dimension.isdigit():
                            continue
                    if "EDGE_WEIGHT_TYPE" in line:
                        EdgeWeightType = line.strip().split()[1]
                        if EdgeWeightType != "EUC_2D":
                            continue
                    if "NODE_COORD_SECTION" in line:
                        x_y = None
                        nodelist_x = []
                        nodelist_y = []
        
                        for j in range (1, int(Dimension)):
                            x_y = lines[i+1].strip().split()[1:]
    #                        print(x_y)
    #                        import pdb; pdb.set_trace()
                            x = x_y[0]
                            y = x_y[1]
                            nodelist_x.append(float(x))
                            nodelist_y.append(float(y))
                            i+=1
                        tsp_database.update({city[:-4]:[nodelist_x, nodelist_y]})
    return tsp_database

tsp_database = gen_tsp_data()
problem_list = tsp_database.keys()

tsp_list = ["berlin52","eil76","eil101","kroE100","kroB200","gil262","lin318","pcb442","rat575","gr666","rat783","pr1002","d2103","u2152","pr2392","pcb3038","fnl4461","rl5915","rl5934","rl11849","pla33810"]
#tsp_list = ["d2103","u2152","pr2392","pcb3038","fnl4461","rl5915","rl5934","rl11849","pla33810"]

for x in tsp_list:
    X_coord = np.array(tsp_database[x][0])
    Y_coord = np.array(tsp_database[x][1])

    with open("tsp" + x + "_pyconcorde.txt", "w") as f:
        tic = time.perf_counter()
        solver = TSPSolver.from_data(X_coord, Y_coord, norm="EUC_2D")  
        solution = solver.solve()
        toc = time.perf_counter()
        f.write( str("Total Time: ") + str(toc-tic))
        f.write( str(" ") + str('output') + str(" ") )
        f.write( str("solution found? ") + str(solution.found_tour) + str(" ") )
        f.write( str("Optimal value? ") + str(solution.optimal_value) + str(" ") )
        f.write( str(" ").join( str(node_idx+1) for node_idx in solution.tour) )
        f.write( str(" ") + str(solution.tour[0]+1) + str(" ") )