import numpy as np
import networkx as nx

adj_graphs = {}
adj_graphs['line'] = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0]
        ])

adj_graphs['circle'] = np.array([
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0]
])

adj_graphs['disconn'] = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0]
])

adj_graphs['btree'] = nx.to_numpy_array(nx.balanced_tree(2,2)).astype(int)
adj_graphs['utree'] = np.array([[0, 1, 1, 0, 0, 0, 0],
                                [1, 0, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0], 
                                [0, 1, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0]])
adj_graphs['grid'] = nx.to_numpy_array(nx.grid_2d_graph(6,6, periodic=False)).astype(int)

adj_graphs['tree9'] = np.zeros((9, 9), dtype=int)
edges_tree9 = [(0,1), (0,2), (1,3), (2,4), (3,5), (3,6), (4,7), (4,8)]
for i, j in edges_tree9:
    adj_graphs['tree9'][i,j] = 1
    adj_graphs['tree9'][j,i] = 1

