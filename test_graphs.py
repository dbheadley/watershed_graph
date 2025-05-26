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

