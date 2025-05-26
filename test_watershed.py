import unittest
import numpy as np
import networkx

from watershed import watershed_g, watershed_gt
from test_graphs import adj_graphs

def interpmv(x_target, xi_known, yi_known_T):
    # x_target: 1D array of target time points (e.g., np.arange(6))
    # xi_known: 1D array of known time points (e.g., np.array([0, 5]))
    # yi_known_T: 2D array (Nodes, KnownTimePoints) of known values (transposed from notebook's yi)
    # Returns: 2D array (TimePoints, Nodes)
    y_interpolated_T_N = np.zeros((x_target.size, yi_known_T.shape[0]))
    for i in range(yi_known_T.shape[0]): # Iterate over nodes
        y_interpolated_T_N[:, i] = np.interp(x_target, xi_known, yi_known_T[i, :])
    return y_interpolated_T_N

class TestWatershedG(unittest.TestCase):
    def test_line_graph_static(self):
        graph_line = adj_graphs['line']
        node_values_list = [
            np.array([-1, -1, -1, -1, 0, 0]),
            np.array([-1, -2, -1, 0, -1, -2]),
            np.array([-1, 0, -1, 0, -1, 0]),
            np.array([-1, -2, -3, -2, -1, 0]),
            np.array([3, 2, 1, 0, -1, -2])
        ]
        expected_labels_list = [
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 2, 2]),
            np.array([1, 1, 2, 2, 3, 3]),
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1])
        ]

        for node_values, expected_labels in zip(node_values_list, expected_labels_list):
            labels = watershed_g(graph_line, node_values)
            np.testing.assert_array_equal(labels, expected_labels)

    def test_circle_graph_static(self):
        graph_circle = adj_graphs['circle']
        node_values_list = [
            np.array([-1, -2, -3, -2, -1, 0]),
            np.array([-1, 0, -1, 0, -1, 0]),
            np.array([-1, -1, -1, -1, 0, 0]),
            np.array([-1, -2, -1, 0, -1, -2])
        ]
        expected_labels_list = [
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 2, 2, 3, 1]),  # Adjusted based on typical watershed behavior with multiple minima
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 2, 2])
        ]

        for node_values, expected_labels in zip(node_values_list, expected_labels_list):
            labels = watershed_g(graph_circle, node_values)
            np.testing.assert_array_equal(labels, expected_labels)

    def test_tree_graph_static(self):
        graph_tree_adj = adj_graphs['tree9']

        node_values_list = [
            np.array([0, 0, -1, 0, -2, 0, 0, -1, -3]),
            np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1]),
            np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9])
        ]
        expected_labels_list = [
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1, 3, 1, 3, 1, 4, 3, 2, 1])
        ]

        for node_values, expected_labels in zip(node_values_list, expected_labels_list):
            labels = watershed_g(graph_tree_adj, node_values)
            np.testing.assert_array_equal(labels, expected_labels)

    def test_grid_graph_static(self):
        graph_grid = adj_graphs['grid'] # 36 nodes
        node_values_list = [
            -np.array([1,2,3,4,5,6, 2,3,4,5,6,7, 3,4,5,6,7,8, 4,5,6,7,8,9, 5,6,7,8,9,10, 6,7,8,9,10,11], dtype=float),
            -np.array([0,1,2,1,0,0, 1,2,3,2,1,0, 2,3,4,3,2,1, 2,3,4,3,2,1, 1,2,3,2,1,0, 0,1,2,1,0,0]),
            -np.array([4,3,2,1,0,0, 3,2,1,0,0,0, 2,1,0,0,0,1, 1,0,0,0,1,2, 0,0,0,1,2,3, 0,0,1,2,3,4])
        ]
        expected_labels_list = [
            np.ones(36, dtype=int),
            np.ones(36, dtype=int),
            np.array([1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,2,2,1,1,1,2,2,2,1,1,2,2,2,2,1,2,2,2,2,2])
        ]

        for node_values, expected_labels in zip(node_values_list, expected_labels_list):
            labels = watershed_g(graph_grid, node_values)
            np.testing.assert_array_equal(labels, expected_labels)

class TestWatershedGT(unittest.TestCase):
    def test_line_graph_temporal(self):
        graph_line = adj_graphs['line']
        xi = np.array([0, 5])
        x_target = np.arange(6)

        # Test Case 1
        yi_known_T_1 = -np.array([[3,2,1,0,0,0], [0,0,0,0,0,0]]).T
        val_tn_1 = interpmv(x_target, xi, yi_known_T_1)
        expected_labels_tn_1 = np.ones((6,6), dtype=int)
        labels_tn_1 = watershed_gt(graph_line, val_tn_1)
        np.testing.assert_array_equal(labels_tn_1, expected_labels_tn_1)

        # Test Case 3
        yi_known_T_3 = -np.array([[3,2,1,0,0,0], [0,0,0,1,2,3]]).T
        val_tn_3 = interpmv(x_target, xi, yi_known_T_3)
        expected_labels_tn_3 = np.array([[1,1,1,1,2,1], [1,1,1,1,2,1], [1,1,1,2,2,1], [1,1,1,2,2,2], [2,1,2,2,2,2], [2,1,2,2,2,2]])
        labels_tn_3 = watershed_gt(graph_line, val_tn_3)
        np.testing.assert_array_equal(labels_tn_3, expected_labels_tn_3)

        # Test Case 4
        yi_known_T_4 = -np.array([[3,2,1,0,0,0], [0,1,2,1,2,3]]).T
        val_tn_4 = interpmv(x_target, xi, yi_known_T_4)
        expected_labels_tn_4 = np.array([[1,1,1,1,2,1], [1,1,1,1,2,1], [1,1,1,1,2,1], [1,1,3,3,2,2], [2,3,3,3,2,2], [2,3,3,3,2,2]])
        labels_tn_4 = watershed_gt(graph_line, val_tn_4)
        np.testing.assert_array_equal(labels_tn_4, expected_labels_tn_4)

        # Test Case 5
        val_tn_5 = -np.eye(6)
        expected_labels_tn_5 = np.ones((6,6), dtype=int)
        labels_tn_5 = watershed_gt(graph_line, val_tn_5)
        np.testing.assert_array_equal(labels_tn_5, expected_labels_tn_5)

        # Test Case 6
        val_tn_6 = -np.eye(6) + -np.eye(6, k=1)
        expected_labels_tn_6 = np.ones((6,6), dtype=int)
        labels_tn_6 = watershed_gt(graph_line, val_tn_6)
        np.testing.assert_array_equal(labels_tn_6, expected_labels_tn_6)

        # Test Case 7
        val_tn_7 = -np.eye(6) + -np.eye(6, k=3)
        expected_labels_tn_7 = np.array([[1,1,2,2,2,1], [1,1,1,2,2,2], [2,1,1,1,2,2], [2,1,1,1,1,2], [1,1,1,1,1,1], [1,1,1,1,1,1]])
        labels_tn_7 = watershed_gt(graph_line, val_tn_7)
        np.testing.assert_array_equal(labels_tn_7, expected_labels_tn_7)

        # Test Case 8
        val_tn_8 = -np.eye(6, k=-3) + -np.eye(6, k=3)
        expected_labels_tn_8 = np.ones((6,6), dtype=int)
        labels_tn_8 = watershed_gt(graph_line, val_tn_8)
        np.testing.assert_array_equal(labels_tn_8, expected_labels_tn_8)

        # Test Case 9
        val_tn_9 = -np.eye(6,k=-3) + -np.eye(6,k=3) + -np.eye(6)
        expected_labels_tn_9 = np.array([[1,1,2,2,2,1], [1,1,1,2,2,2], [2,1,1,1,2,2], [2,2,1,1,1,2], [2,2,2,1,1,1], [1,2,2,2,1,1]])
        labels_tn_9 = watershed_gt(graph_line, val_tn_9)
        np.testing.assert_array_equal(labels_tn_9, expected_labels_tn_9)

        # Test Case 10
        val_tn_10 = np.zeros((6,6))
        val_tn_10[2,3] = -1
        expected_labels_tn_10 = np.ones((6,6), dtype=int)
        labels_tn_10 = watershed_gt(graph_line, val_tn_10)
        np.testing.assert_array_equal(labels_tn_10, expected_labels_tn_10)

        # Test Case 11
        val_tn_11 = -np.array([[1,0,0,0,0,0], [0,2,0,0,0,0], [0,0,3,0,0,0], [0,0,0,4,0,0], [0,0,0,0,5,0], [0,0,0,0,0,6]])
        expected_labels_tn_11 = np.ones((6,6), dtype=int)
        labels_tn_11 = watershed_gt(graph_line, val_tn_11)
        np.testing.assert_array_equal(labels_tn_11, expected_labels_tn_11)

        # Test Case 12
        val_tn_12 = -np.array([[1,0,0,0,0,0], [0,2,0,0,0,0], [0,0,3,0,0,0], [0,0,0,4,0,0], [0,0,0,0,5,0], [0,0,0,0,0,6]])
        val_tn_12[1,1] = -3
        expected_labels_tn_12 = np.ones((6,6), dtype=int)
        labels_tn_12 = watershed_gt(graph_line, val_tn_12)
        np.testing.assert_array_equal(labels_tn_12, expected_labels_tn_12)

        # Test Case 13
        val_tn_13 = np.rot90(-np.array([[1,0,0,0,0,0], [0,2,0,0,0,0], [0,0,3,0,0,0], [0,0,0,4,0,0], [0,0,0,0,5,0], [0,0,0,0,0,6]]))
        expected_labels_tn_13 = np.ones((6,6), dtype=int)
        labels_tn_13 = watershed_gt(graph_line, val_tn_13)
        np.testing.assert_array_equal(labels_tn_13, expected_labels_tn_13)

        # Test Case 14
        val_tn_14 = -np.array([[6,0,0,0,0,0], [0,5,0,0,0,0], [0,0,4,0,0,0], [0,0,0,3,0,0], [0,0,0,0,2,0], [0,0,0,0,0,1]])
        expected_labels_tn_14 = np.ones((6,6), dtype=int)
        labels_tn_14 = watershed_gt(graph_line, val_tn_14)
        np.testing.assert_array_equal(labels_tn_14, expected_labels_tn_14)

        # Test Case 15
        val_tn_15 = -np.array([[1,2,0,0,0,0], [0,2,3,0,0,0], [0,0,3,4,0,0], [0,0,0,4,5,0], [0,0,0,0,5,6], [0,0,0,0,0,6]])
        expected_labels_tn_15 = np.ones((6,6), dtype=int)
        labels_tn_15 = watershed_gt(graph_line, val_tn_15)
        np.testing.assert_array_equal(labels_tn_15, expected_labels_tn_15)

    def test_circle_graph_temporal(self):
        graph_circle = adj_graphs['circle']
        xi = np.array([0, 5])
        x_target = np.arange(6)

        # Test Case Circ_1
        yi_known_T_1 = -np.array([[3,2,1,0,0,0], [0,0,0,0,0,0]]).T
        val_tn_1 = interpmv(x_target, xi, yi_known_T_1)
        expected_labels_tn_1 = np.ones((6,6), dtype=int)
        labels_tn_1 = watershed_gt(graph_circle, val_tn_1)
        np.testing.assert_array_equal(labels_tn_1, expected_labels_tn_1)

        # Test Case Circ_2
        yi_known_T_2 = -np.array([[3,2,1,0,0,0], [0,0,0,1,2,3]]).T
        val_tn_2 = interpmv(x_target, xi, yi_known_T_2)
        expected_labels_tn_2 = np.array([[1,1,1,1,2,1], [1,1,1,1,2,1], [1,1,1,2,2,1], [1,1,1,2,2,1], [2,1,2,2,2,1], [2,1,2,2,2,1]])
        labels_tn_2 = watershed_gt(graph_circle, val_tn_2)
        np.testing.assert_array_equal(labels_tn_2, expected_labels_tn_2)

        # Test Case Circ_3
        val_tn_3 = -np.eye(6)
        expected_labels_tn_3 = np.ones((6,6), dtype=int)
        labels_tn_3 = watershed_gt(graph_circle, val_tn_3)
        np.testing.assert_array_equal(labels_tn_3, expected_labels_tn_3)

        # Test Case Circ_4
        val_tn_4 = -np.eye(6) + -np.eye(6, k=3)
        expected_labels_tn_4 = np.array([[1,1,2,2,2,1], [1,1,1,2,2,1], [2,1,1,1,2,2], [2,2,1,1,1,2], [2,2,2,1,1,1], [1,2,2,2,1,1]])
        labels_tn_4 = watershed_gt(graph_circle, val_tn_4)
        np.testing.assert_array_equal(labels_tn_4, expected_labels_tn_4)

    def test_branched_graph_temporal(self):
        graph_utree = adj_graphs['utree'] # 7 nodes
        xi = np.array([0, 5]) # For Test Case Branch_4_simple
        x_target = np.arange(6) # For Test Case Branch_4_simple

        # Test Case Branch_1
        val_tn_1 = np.zeros((6,7))
        val_tn_1[:,0] = -1
        expected_labels_tn_1 = np.ones((6,7), dtype=int)
        labels_tn_1 = watershed_gt(graph_utree, val_tn_1)
        np.testing.assert_array_equal(labels_tn_1, expected_labels_tn_1)

        # Test Case Branch_2
        val_tn_2 = np.zeros((6,7))
        val_tn_2[:,3] = -1
        expected_labels_tn_2 = np.ones((6,7), dtype=int)
        labels_tn_2 = watershed_gt(graph_utree, val_tn_2)
        np.testing.assert_array_equal(labels_tn_2, expected_labels_tn_2)

        # Test Case Branch_3
        val_tn_3 = np.zeros((6,7))
        val_tn_3[:,[1,3]] = -1
        expected_labels_tn_3 = np.ones((6,7), dtype=int)
        labels_tn_3 = watershed_gt(graph_utree, val_tn_3)
        np.testing.assert_array_equal(labels_tn_3, expected_labels_tn_3)

        # Test Case Branch_4 (simplified)
        yi_known_T_branch = -np.array([[3,2,1,0,0,0,0], [0,0,0,0,0,0,0]]).T # 7 nodes
        val_tn_branch_4_simple = interpmv(x_target, xi, yi_known_T_branch)
        expected_labels_tn_branch_4_simple = np.ones((6,7), dtype=int)
        labels_tn_4 = watershed_gt(graph_utree, val_tn_branch_4_simple)
        np.testing.assert_array_equal(labels_tn_4, expected_labels_tn_branch_4_simple)

if __name__ == '__main__':
    unittest.main()
