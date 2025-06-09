import unittest
import numpy as np
import networkx

from watershed import watershed_g, watershed_gt
from test_graphs import adj_graphs

def interpmv(x, xi, yi):
    """
    Interpolate the values of yi at the points xi to the points x.

    Parameters
    ----------
    x : array-like, shape (n)
        The points at which to interpolate the values.
    xi : array-like, shape (n, m)
        The points at which the values are known.
    yi : array-like, shape (n, m)
        The values at the points xi.
    Returns
    -------
    y : array-like
        The interpolated values at the points x.
    """
    y = np.zeros((x.size, yi.shape[1]))
    for i in range(yi.shape[1]): # Iterate over nodes
        y[:, i] = np.interp(x, xi, yi[:, i])
    return y

def graph_flow(graph, val_list):
    """
    Flow the values of the graph along its edges.

    Parameters
    ----------
    graph : array-like, shape (n, n)
        The graph to flow the values along.
    val_list : list of array-like, shape (n)
        The values to flow along the graph.
    Returns
    -------
    y : array-like, shape (n, m)
        The flowed values at the points x.
    """
    y = np.zeros(val_list.shape)
    graph_n = graph/np.sum(graph, axis=1)
    y[0,:] = val_list[0, :]
    for i in range(1,val_list.shape[0]):
        y[i,:] = y[i-1,:]@graph_n + val_list[i,:]
    return y

def run_test_set_g(graph, values, labels):
    """
    Run a test set for the watershed_g function.

    Parameters
    ----------
    graph : array-like, shape (n, n)
        The graph to test.
    values : list of array-like, shape (n)
        The values to test.
    labels : list of array-like, shape (n)
        The expected labels for the values.
    """
    for ind, (val, label) in enumerate(zip(values, labels)):
        result = watershed_g(graph, val)

        err_msg = f"Test {ind} failed for values: {val}\n Expected: {label}\n Got: {result}"
        np.testing.assert_array_equal(result, label, err_msg=err_msg)

def run_test_set_gt(graph, values, labels):
    """
    Run a test set for the watershed_gt function.

    Parameters
    ----------
    graph : array-like, shape (n, n)
        The graph to test.
    values : list of array-like, shape (n, m)
        The values to test.
    labels : list of array-like, shape (n, m)
        The expected labels for the values.
    """
    for ind, (val, label) in enumerate(zip(values, labels)):
        result = watershed_gt(graph, val)

        err_msg = f"Test {ind} failed for values: {val}\n Expected: {label}\n Got: {result}"
        np.testing.assert_array_equal(result, label, err_msg=err_msg)

class TestWatershedG(unittest.TestCase):
    def test_line_graph_static(self):
        graph_line = adj_graphs['line']
        node_values_list = [
            -np.array([1, 1, 1, 1, 0, 0]),
            -np.array([1, 2, 1, 0, 1, 2]),
            -np.array([1, 0, 1, 0, 1, 0]),
            -np.array([1, 2, 3, 2, 1, 0]),
            -np.array([-3, -2, -1, 0, 1, 2])
        ]
        expected_labels_list = [
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 2, 2]),
            np.array([1, 1, 2, 2, 3, 3]),
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1])
        ]

        run_test_set_g(graph_line, node_values_list, expected_labels_list)

    def test_circle_graph_static(self):
        graph_circle = adj_graphs['circle']
        node_values_list = [
            -np.array([1, 2, 3, 2, 1, 0]),
            -np.array([1, 0, 1, 0, 1, 0]),
            -np.array([1, 1, 1, 1, 0, 0]),
            -np.array([1, 2, 1, 0, 1, 2])
        ]
        expected_labels_list = [
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 2, 2, 3, 1]),
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 2, 2])
        ]

        run_test_set_g(graph_circle, node_values_list, expected_labels_list)

    def test_tree_graph_static(self):
        graph_tree_adj = adj_graphs['tree9']

        node_values_list = [
            -np.array([0, 0, 1, 0, 2, 0, 0, 1, 3]),
            -np.array([9, 8, 7, 6, 5, 4, 3, 2, 1]),
            -np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ]
        expected_labels_list = [
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1, 3, 1, 3, 1, 4, 3, 2, 1])
        ]

        run_test_set_g(graph_tree_adj, node_values_list, expected_labels_list)

    def test_grid_graph_static(self):
        graph_grid = adj_graphs['grid'] # 36 nodes
        node_values_list = [
            -np.array([1,2,3,4,5,6, 
                       2,3,4,5,6,7, 
                       3,4,5,6,7,8, 
                       4,5,6,7,8,9, 
                       5,6,7,8,9,10, 
                       6,7,8,9,10,11], dtype=float),
            -np.array([0,1,2,1,0,0, 
                       1,2,3,2,1,0, 
                       2,3,4,3,2,1, 
                       2,3,4,3,2,1, 
                       1,2,3,2,1,0, 
                       0,1,2,1,0,0]),
            -np.array([4,3,2,1,0,0, 
                       3,2,1,0,0,0, 
                       2,1,0,0,0,1, 
                       1,0,0,0,1,2, 
                       0,0,0,1,2,3, 
                       0,0,1,2,3,4])
        ]
        expected_labels_list = [
            np.ones(36, dtype=int),
            np.ones(36, dtype=int),
            np.array([1,1,1,1,1,1,
                      1,1,1,1,1,2,
                      1,1,1,1,2,2,
                      1,1,1,2,2,2,
                      1,1,2,2,2,2,
                      1,2,2,2,2,2])
        ]

        run_test_set_g(graph_grid, node_values_list, expected_labels_list)

class TestWatershedGT(unittest.TestCase):
    def test_line_graph_temporal(self):
        graph_line = adj_graphs['line']
        xi = np.array([0, 5])
        x = np.arange(6)

        node_values_list = []
        expected_labels_list = []

        # Test Case 1
        yi_1 = -np.array([[3,2,1,0,0,0], [0,0,0,0,0,0]])
        node_values_list.append(interpmv(x, xi, yi_1))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 2
        node_values_list.append(node_values_list[-1].T)
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 3
        yi_3 = -np.array([[3,2,1,0,0,0], [0,0,0,1,2,3]])
        node_values_list.append(interpmv(x, xi, yi_3))
        expected_labels_list.append(np.array([[1,1,1,1,2,2], 
                                         [1,1,1,1,2,2], 
                                         [1,1,1,2,2,2], 
                                         [1,1,1,2,2,2], 
                                         [1,1,2,2,2,2], 
                                         [1,1,2,2,2,2]]))

        # Test Case 4
        yi_4 = -np.array([[3,2,1,0,0,0], [0,1,2,1,2,3]])
        node_values_list.append(interpmv(x, xi, yi_4))
        expected_labels_list.append(np.array([[1,1,1,1,2,2], 
                                         [1,1,1,1,2,2], 
                                         [1,1,1,1,2,2], 
                                         [1,1,3,3,2,2], 
                                         [1,3,3,3,2,2], 
                                         [3,3,3,3,2,2]]))

        # Test Case 5
        node_values_list.append(-np.eye(6))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 6
        node_values_list.append(-np.eye(6) - np.eye(6, k=1))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 7
        node_values_list.append(-np.eye(6) - np.eye(6, k=3))
        expected_labels_list.append(np.array([[1,1,2,2,2,2], 
                                         [1,1,1,2,2,2], 
                                         [1,1,1,1,2,2], 
                                         [1,1,1,1,1,2], 
                                         [1,1,1,1,1,1], 
                                         [1,1,1,1,1,1]]))

        # Test Case 8
        node_values_list.append(-np.eye(6, k=-3) - np.eye(6, k=3))
        expected_labels_list.append(np.array([[1,1,1,1,1,1], 
                                         [2,1,1,1,1,1], 
                                         [2,2,1,1,1,1], 
                                         [2,2,2,1,1,1], 
                                         [2,2,2,2,1,1], 
                                         [2,2,2,2,2,1]]))

        # Test Case 9
        node_values_list.append(-np.eye(6,k=-3) - np.eye(6,k=3) - np.eye(6))
        expected_labels_list.append(np.array([[1,1,2,2,2,2], 
                                         [1,1,1,2,2,2], 
                                         [3,1,1,1,2,2], 
                                         [3,3,1,1,1,2], 
                                         [3,3,3,1,1,1], 
                                         [3,3,3,3,1,1]]))

        # Test Case 10
        node_values_list.append(-np.array([[0,0,0,1,0,0],
                               [0,0,1,0,1,0],
                               [0,1,0,0,0,1],
                               [0,0,1,0,1,0],
                               [0,0,0,1,0,0],
                               [0,0,0,0,0,0]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 11
        node_values_list.append(-np.array([[1,0,0,0,0,0], 
                               [0,2,0,0,0,0], 
                               [0,0,3,0,0,0], 
                               [0,0,0,4,0,0], 
                               [0,0,0,0,5,0], 
                               [0,0,0,0,0,6]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 12
        node_values_list.append(-np.array([[1,0,0,0,0,0], 
                               [0,3,0,0,0,0], 
                               [0,0,3,0,0,0], 
                               [0,0,0,4,0,0], 
                               [0,0,0,0,5,0], 
                               [0,0,0,0,0,6]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 13
        node_values_list.append(np.rot90(-np.array([[1,0,0,0,0,0], 
                                        [0,2,0,0,0,0], 
                                        [0,0,3,0,0,0], 
                                        [0,0,0,4,0,0], 
                                        [0,0,0,0,5,0], 
                                        [0,0,0,0,0,6]])))
        expected_labels_list.append(np.ones((6,6), dtype=int))
       
       
        # Test Case 14
        node_values_list.append(-np.array([[6,0,0,0,0,0], 
                               [0,5,0,0,0,0], 
                               [0,0,4,0,0,0], 
                               [0,0,0,3,0,0], 
                               [0,0,0,0,2,0], 
                               [0,0,0,0,0,1]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 15
        node_values_list.append(-np.array([[1,2,0,0,0,0], 
                               [0,2,3,0,0,0], 
                               [0,0,3,4,0,0], 
                               [0,0,0,4,5,0], 
                               [0,0,0,0,5,6], 
                               [0,0,0,0,0,6]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        run_test_set_gt(graph_line, node_values_list, expected_labels_list)

    # def test_circle_graph_temporal(self):
    def test_circle_graph_temporal(self):
        graph_line = adj_graphs['circle']
        xi = np.array([0, 5])
        x = np.arange(6)

        node_values_list = []
        expected_labels_list = []

        # Test Case 1
        yi_1 = -np.array([[3,2,1,0,0,0], [0,0,0,0,0,0]])
        node_values_list.append(interpmv(x, xi, yi_1))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 2
        node_values_list.append(node_values_list[-1].T)
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 3
        yi_3 = -np.array([[3,2,1,0,0,0], [0,0,0,1,2,3]])
        node_values_list.append(interpmv(x, xi, yi_3))
        expected_labels_list.append(np.array([[1,1,1,1,2,1], 
                                         [1,1,1,1,2,1], 
                                         [1,1,1,2,2,1], 
                                         [1,1,1,2,2,2], 
                                         [2,1,2,2,2,2], 
                                         [2,1,2,2,2,2]]))

        # Test Case 4
        yi_4 = -np.array([[3,2,1,0,0,0], [0,1,2,1,2,3]])
        node_values_list.append(interpmv(x, xi, yi_4))
        expected_labels_list.append(np.array([[1,1,1,1,2,1], 
                                         [1,1,1,1,2,1], 
                                         [1,1,1,1,2,1], 
                                         [1,1,3,3,2,2], 
                                         [2,3,3,3,2,2], 
                                         [2,3,3,3,2,2]]))

        # Test Case 5
        node_values_list.append(-np.eye(6))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 6
        node_values_list.append(-np.eye(6) - np.eye(6, k=1))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 7
        node_values_list.append(-np.eye(6) - np.eye(6, k=3))
        expected_labels_list.append(np.array([[1,1,2,2,2,1], 
                                         [1,1,1,2,2,2], 
                                         [2,1,1,1,2,2], 
                                         [2,1,1,1,1,2], 
                                         [1,1,1,1,1,1], 
                                         [1,1,1,1,1,1]]))

        # Test Case 8
        node_values_list.append(-np.eye(6, k=-3) - np.eye(6, k=3))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 9
        node_values_list.append(-np.eye(6,k=-3) - np.eye(6,k=3) - np.eye(6))
        expected_labels_list.append(np.array([[1,1,2,2,2,1], 
                                         [1,1,1,2,2,2], 
                                         [2,1,1,1,2,2], 
                                         [2,2,1,1,1,2], 
                                         [2,2,2,1,1,1], 
                                         [1,2,2,2,1,1]]))

        # Test Case 10
        node_values_list.append(-np.array([[0,0,0,1,0,0],
                               [0,0,1,0,1,0],
                               [0,1,0,0,0,1],
                               [0,0,1,0,1,0],
                               [0,0,0,1,0,0],
                               [0,0,0,0,0,0]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 11
        node_values_list.append(-np.array([[1,0,0,0,0,0], 
                               [0,2,0,0,0,0], 
                               [0,0,3,0,0,0], 
                               [0,0,0,4,0,0], 
                               [0,0,0,0,5,0], 
                               [0,0,0,0,0,6]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 12
        node_values_list.append(-np.array([[1,0,0,0,0,0], 
                               [0,3,0,0,0,0], 
                               [0,0,3,0,0,0], 
                               [0,0,0,4,0,0], 
                               [0,0,0,0,5,0], 
                               [0,0,0,0,0,6]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 13
        node_values_list.append(np.rot90(-np.array([[1,0,0,0,0,0], 
                                        [0,2,0,0,0,0], 
                                        [0,0,3,0,0,0], 
                                        [0,0,0,4,0,0], 
                                        [0,0,0,0,5,0], 
                                        [0,0,0,0,0,6]])))
        expected_labels_list.append(np.ones((6,6), dtype=int))
       
       
        # Test Case 14
        node_values_list.append(-np.array([[6,0,0,0,0,0], 
                               [0,5,0,0,0,0], 
                               [0,0,4,0,0,0], 
                               [0,0,0,3,0,0], 
                               [0,0,0,0,2,0], 
                               [0,0,0,0,0,1]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        # Test Case 15
        node_values_list.append(-np.array([[1,2,0,0,0,0], 
                               [0,2,3,0,0,0], 
                               [0,0,3,4,0,0], 
                               [0,0,0,4,5,0], 
                               [0,0,0,0,5,6], 
                               [0,0,0,0,0,6]]))
        expected_labels_list.append(np.ones((6,6), dtype=int))

        run_test_set_gt(graph_line, node_values_list, expected_labels_list)

    def test_temporal_branch_graph(self):
        graph_branch = np.zeros((7, 7))
        graph_branch[0, 1] = 1; graph_branch[1, 0] = 1
        graph_branch[1, 2] = 1; graph_branch[2, 1] = 1
        graph_branch[1, 3] = 1; graph_branch[3, 1] = 1
        graph_branch[3, 4] = 1; graph_branch[4, 3] = 1
        graph_branch[3, 5] = 1; graph_branch[5, 3] = 1
        graph_branch[3, 6] = 1; graph_branch[6, 3] = 1

        node_values_list = []
        expected_labels_list = []

        # Test Case 1: Activate single node at binary branch
        node_values = np.zeros((6, 7))
        node_values[:, 0] = -1
        node_values_list.append(graph_flow(graph_branch, node_values))
        expected_labels_list.append(np.ones((6, 7), dtype=int))

        # Test Case 2: Activate single node at trinary branch
        node_values = np.zeros((6, 7))
        node_values[:, 3] = -1
        node_values_list.append(node_values)
        expected_labels_list.append(np.ones((6, 7), dtype=int))

        # Test Case 3: Activate two nodes in the center
        node_values = np.zeros((6, 7))
        node_values[:, [1, 3]] = -1
        node_values_list.append(node_values)
        expected_labels_list.append(np.ones((6, 7), dtype=int))

        # Test Case 4: Activate two nodes at edges, switching from one set to the other
        node_values = np.zeros((6, 7))
        node_values[0, [2, 6]] = -1
        node_values[-1, [0, 4]] = -1
        node_values_list.append(graph_flow(graph_branch, node_values))
        expected_labels_list.append(np.array([[3,3,3,4,4,4,4],
                                              [1,3,3,3,1,1,4],
                                              [1,1,1,1,1,1,1],
                                              [1,1,1,1,2,2,2],
                                              [1,1,1,2,2,2,2],
                                              [1,1,1,2,2,2,2]]))

        # Test Case 5: Activate each edge node one at a time
        node_values = np.zeros((6, 7))
        node_values[0, 2] = -1
        node_values[1, 4] = -1
        node_values[2, 5] = -1
        node_values[3, 6] = -1
        node_values[4, 0] = -1
        node_values_list.append(node_values)
        expected_labels_list.append(np.array([[1,1,1,1,2,3,1],
                                              [1,1,1,2,2,3,2],
                                              [5,3,1,3,2,3,4],
                                              [5,4,1,4,2,3,4],
                                              [5,5,5,4,2,3,4],
                                              [5,5,5,4,2,3,4]]))

        # Test Case 6: Move activation from one end of the tree to the other
        node_values = np.zeros((6, 7))
        node_values[0, 0] = -1
        node_values[1, 1] = -1
        node_values[2, 2] = -1
        node_values[3, 1] = -1
        node_values[4, 3] = -1
        node_values[5, 4] = -1
        node_values_list.append(node_values)
        expected_labels_list.append(np.ones((6, 7), dtype=int))

        # Test Case 7: Move activation with varying amplitude
        node_values = np.zeros((6, 7))
        node_values[0, 0] = -1
        node_values[1, 1] = -2
        node_values[2, 2] = -3
        node_values[3, 1] = -4
        node_values[4, 3] = -3
        node_values[5, 4] = -2
        node_values_list.append(node_values)
        expected_labels_list.append(np.ones((6, 7), dtype=int))

        run_test_set_gt(graph_branch, node_values_list, expected_labels_list)
        
if __name__ == '__main__':
    unittest.main()
