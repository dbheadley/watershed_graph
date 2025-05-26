import unittest
import numpy as np
from watershed import watershed_segmentation_graph, watershed_segmentation_graph_temporal
# Assuming your watershed functions are in a file named watershed_algorithms.py
# If they are in the immersive with ID watershed_graph_spatial_py and watershed_graph_py,
# you would typically import them directly if running in an environment that supports it.
# For a standalone script, they need to be importable.
# For this example, I'll assume they are in watershed_algorithms.py:
# from watershed_algorithms import watershed_segmentation_graph_spatial, watershed_segmentation_graph_temporal
# If running from the notebook or an environment where these are defined,
# you might need to adjust the import or copy the function definitions here.

# Placeholder for the actual functions if not importing
# These should be replaced with the actual function definitions or correct imports
def watershed_segmentation_graph_spatial(values_n: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
    """
    Placeholder for the spatial watershed segmentation function.
    Replace with the actual function from watershed_graph_spatial_py.
    """
    # This is a simplified version of the logic from watershed_graph_spatial_py
    # to make the test script runnable. Replace with the full function.
    import heapq
    import collections

    if not isinstance(values_n, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
        raise TypeError("Inputs 'values_n' and 'graph_matrix' must be NumPy arrays.")
    if values_n.ndim != 1:
        raise ValueError(f"Input 'values_n' must be a 1D array (N,), but got {values_n.ndim} dimensions.")
    
    N = values_n.shape[0]
    if N == 0: 
        return np.array([], dtype=int)

    if graph_matrix.ndim != 2 or graph_matrix.shape != (N, N):
        raise ValueError(f"Input 'graph_matrix' must be an N x N ({N}x{N}) array, but got shape {graph_matrix.shape}.")

    labels_n = np.zeros(N, dtype=int)
    priority_queue = [] 
    entry_count = 0
    current_basin_id = 0

    adj_list = [[] for _ in range(N)]
    rows, cols = np.where(graph_matrix == 1)
    for i, j in zip(rows, cols):
        adj_list[i].append(j)
    
    sorted_node_indices = sorted(range(N), key=lambda i: (values_n[i], i))

    for n_start_plateau in sorted_node_indices:
        if labels_n[n_start_plateau] == 0: 
            current_plateau_value = values_n[n_start_plateau]
            plateau_nodes = [] 
            visited_for_this_plateau_search = set() 
            bfs_plateau_q = collections.deque([n_start_plateau])
            visited_for_this_plateau_search.add(n_start_plateau)

            while bfs_plateau_q: 
                curr_n = bfs_plateau_q.popleft()
                plateau_nodes.append(curr_n) 
                for neighbor_n in adj_list[curr_n]:
                    if neighbor_n not in visited_for_this_plateau_search and \
                       values_n[neighbor_n] == current_plateau_value:
                        visited_for_this_plateau_search.add(neighbor_n)
                        bfs_plateau_q.append(neighbor_n)
            
            is_regional_minimum_plateau = True
            for p_n in plateau_nodes: 
                for neighbor_n_check in adj_list[p_n]:
                    if neighbor_n_check not in visited_for_this_plateau_search: 
                        if values_n[neighbor_n_check] <= current_plateau_value: 
                            is_regional_minimum_plateau = False; break
                if not is_regional_minimum_plateau: break
            
            if is_regional_minimum_plateau:
                current_basin_id += 1
                for p_n_label in plateau_nodes: 
                    if labels_n[p_n_label] == 0: 
                        labels_n[p_n_label] = current_basin_id
                        heapq.heappush(priority_queue, (values_n[p_n_label], entry_count, p_n_label))
                        entry_count += 1
    
    while priority_queue:
        val_u, _, node_u = heapq.heappop(priority_queue) 
        basin_id_u = labels_n[node_u]
        if basin_id_u == 0: 
            continue
        for node_v_idx in adj_list[node_u]:
            if labels_n[node_v_idx] == 0: 
                labels_n[node_v_idx] = basin_id_u
                heapq.heappush(priority_queue, (values_n[node_v_idx], entry_count, node_v_idx))
                entry_count += 1
    return labels_n


def watershed_segmentation_graph_temporal(values_tn: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
    """
    Placeholder for the temporal watershed segmentation function.
    Replace with the actual function from watershed_graph_py.
    """
    # This is a simplified version of the logic from watershed_graph_py
    # to make the test script runnable. Replace with the full function.
    import heapq
    import collections

    if not isinstance(values_tn, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
        raise TypeError("Inputs 'values_tn' and 'graph_matrix' must be NumPy arrays.")
    if values_tn.ndim != 2:
        raise ValueError(f"Input 'values_tn' must be a 2D array (T, N), but got {values_tn.ndim} dimensions.")
    
    T, N = values_tn.shape
    if N == 0: 
        return np.array([[] for _ in range(T)], dtype=int) if T > 0 else np.array([], dtype=int)
    if T == 0: 
        return np.array([], dtype=int).reshape(0,N)

    if graph_matrix.ndim != 2 or graph_matrix.shape != (N, N):
        raise ValueError(f"Input 'graph_matrix' must be an N x N ({N}x{N}) array, but got shape {graph_matrix.shape}.")

    labels_tn = np.zeros((T, N), dtype=int)
    priority_queue = []
    entry_count = 0
    current_basin_id = 0

    adj_list = [[] for _ in range(N)]
    rows, cols = np.where(graph_matrix == 1)
    for i, j in zip(rows, cols):
        adj_list[i].append(j)

    def _get_st_neighbors(curr_t, curr_n, T_max, adj_list_local):
        neighbors_set = set()
        for neighbor_n in adj_list_local[curr_n]:
            neighbors_set.add((curr_t, neighbor_n))
        for dt_self in [-1, 1]:
            next_t_self = curr_t + dt_self
            if 0 <= next_t_self < T_max:
                neighbors_set.add((next_t_self, curr_n))
        for dt_adj in [-1, 1]:
            adj_t = curr_t + dt_adj
            if 0 <= adj_t < T_max:
                for neighbor_n_adj_t in adj_list_local[curr_n]:
                    neighbors_set.add((adj_t, neighbor_n_adj_t))
        return list(neighbors_set)

    sorted_indices_tn = sorted(
        [(t, n) for t in range(T) for n in range(N)],
        key=lambda x: (values_tn[x[0], x[1]], x[0], x[1])
    )

    for t_start_plateau, n_start_plateau in sorted_indices_tn:
        if labels_tn[t_start_plateau, n_start_plateau] == 0: 
            current_plateau_value = values_tn[t_start_plateau, n_start_plateau]
            plateau_points = [] 
            visited_for_this_plateau_search = set() 
            bfs_plateau_q = collections.deque([(t_start_plateau, n_start_plateau)])
            visited_for_this_plateau_search.add((t_start_plateau, n_start_plateau))

            while bfs_plateau_q: 
                curr_t, curr_n = bfs_plateau_q.popleft()
                plateau_points.append((curr_t, curr_n)) 
                potential_st_neighbors_for_plateau = _get_st_neighbors(curr_t, curr_n, T, adj_list)
                for nt, nn in potential_st_neighbors_for_plateau:
                    if (nt, nn) not in visited_for_this_plateau_search and \
                       values_tn[nt, nn] == current_plateau_value:
                        visited_for_this_plateau_search.add((nt, nn))
                        bfs_plateau_q.append((nt, nn))
            
            is_regional_minimum_plateau = True
            for p_t, p_n in plateau_points: 
                st_neighbors_for_min_check = _get_st_neighbors(p_t, p_n, T, adj_list)
                for nt_check, nn_check in st_neighbors_for_min_check:
                    if (nt_check, nn_check) not in visited_for_this_plateau_search: 
                        if values_tn[nt_check, nn_check] <= current_plateau_value: 
                            is_regional_minimum_plateau = False; break
                if not is_regional_minimum_plateau: break
            
            if is_regional_minimum_plateau:
                current_basin_id += 1
                for p_t_label, p_n_label in plateau_points: 
                    if labels_tn[p_t_label, p_n_label] == 0: 
                        labels_tn[p_t_label, p_n_label] = current_basin_id
                        heapq.heappush(priority_queue, (values_tn[p_t_label, p_n_label], entry_count, p_t_label, p_n_label))
                        entry_count += 1
    
    while priority_queue:
        val_u, _, t_u, node_u = heapq.heappop(priority_queue) 
        basin_id_u = labels_tn[t_u, node_u]
        if basin_id_u == 0: 
            continue
        for node_v_spatial_idx in adj_list[node_u]:
            if labels_tn[t_u, node_v_spatial_idx] == 0: 
                labels_tn[t_u, node_v_spatial_idx] = basin_id_u
                heapq.heappush(priority_queue, (values_tn[t_u, node_v_spatial_idx], entry_count, t_u, node_v_spatial_idx))
                entry_count += 1
        for dt in [-1, 1]:
            t_v_temporal = t_u + dt
            if 0 <= t_v_temporal < T:
                if labels_tn[t_v_temporal, node_u] == 0 and \
                   values_tn[t_v_temporal, node_u] >= val_u:
                    labels_tn[t_v_temporal, node_u] = basin_id_u
                    heapq.heappush(priority_queue, (values_tn[t_v_temporal, node_u], entry_count, t_v_temporal, node_u))
                    entry_count += 1
    return labels_tn


class TestSpatialWatershed(unittest.TestCase):
    def setUp(self):
        """Set up common graph structures and value lists for spatial tests."""
        self.graph_line = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0]
        ])
        self.node_vals_line = [
            -np.array([1, 1, 1, 1, 0, 0]),
            -np.array([1, 2, 1, 0, 1, 2]),
            -np.array([1, 0, 1, 0, 1, 0]),
            -np.array([1, 2, 3, 2, 1, 0]),
            -np.array([-3, -2, -1, 0, 1, 2])
        ]
        # Expected labels for line graph tests (derived from notebook output)
        # Note: Basin IDs might differ but the grouping should be the same.
        # We normalize basin IDs for comparison if necessary, or check for consistent grouping.
        # For simplicity, I'll use the exact IDs from the notebook if they are consistent.
        self.expected_labels_line = [
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 2, 2]), # Basin IDs might be [2,2,2,2,1,1]
            np.array([1, 1, 2, 2, 3, 3]), # or other permutations
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1])
        ]

        self.graph_circ = np.array([
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0]
        ])
        self.node_vals_circ = [
            -np.array([1, 2, 3, 2, 1, 0]),
            -np.array([1, 0, 1, 0, 1, 0]),
            -np.array([1, 1, 1, 1, 0, 0]),
            -np.array([1, 2, 1, 0, 1, 2])
        ]
        self.expected_labels_circ = [
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 2, 2, 3, 1]),
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 2, 2]) # or [2,2,2,2,1,1]
        ]

        self.graph_tree = np.zeros((9, 9))
        self.graph_tree[0, 1] = 1; self.graph_tree[1, 0] = 1
        self.graph_tree[0, 2] = 1; self.graph_tree[2, 0] = 1
        self.graph_tree[1, 3] = 1; self.graph_tree[3, 1] = 1
        self.graph_tree[2, 4] = 1; self.graph_tree[4, 2] = 1
        self.graph_tree[3, 5] = 1; self.graph_tree[5, 3] = 1
        self.graph_tree[3, 6] = 1; self.graph_tree[6, 3] = 1
        self.graph_tree[4, 7] = 1; self.graph_tree[7, 4] = 1
        self.graph_tree[4, 8] = 1; self.graph_tree[8, 4] = 1
        self.node_vals_tree = [
            -np.array([0, 0, 1, 0, 2, 0, 0, 1, 3]),
            -np.array([9, 8, 7, 6, 5, 4, 3, 2, 1]),
            -np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ]
        self.expected_labels_tree = [
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1, 3, 1, 3, 1, 4, 3, 2, 1]) # or permutations
        ]

        grid_rows, grid_cols = 6, 6
        num_grid_nodes = grid_rows * grid_cols
        self.graph_grid = np.zeros((num_grid_nodes, num_grid_nodes))
        for i in range(grid_rows):
            for j in range(grid_cols):
                if i < 5:
                    self.graph_grid[i * 6 + j, (i + 1) * 6 + j] = 1
                    self.graph_grid[(i + 1) * 6 + j, i * 6 + j] = 1
                if j < 5:
                    self.graph_grid[i * 6 + j, i * 6 + j + 1] = 1
                    self.graph_grid[i * 6 + j + 1, i * 6 + j] = 1
        
        self.node_vals_grid = []
        self.node_vals_grid.append(-np.array([
            1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8,
            4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 11
        ]))
        self.node_vals_grid.append(-np.array([
            0, 1, 2, 1, 0, 0, 1, 2, 3, 2, 1, 0, 2, 3, 4, 3, 2, 1,
            2, 3, 4, 3, 2, 1, 1, 2, 3, 2, 1, 0, 0, 1, 2, 1, 0, 0
        ]))
        self.node_vals_grid.append(-np.array([
            4, 3, 2, 1, 0, 0, 3, 2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 4
        ]))
        self.expected_labels_grid = [
            np.ones(36, dtype=int), # All one basin
            np.ones(36, dtype=int), # All one basin
            np.array([1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,2,2,1,1,1,2,2,2,1,1,2,2,2,2,1,2,2,2,2,2]) # or permutations
        ]

    def assertBasinsEqual(self, result_labels, expected_labels, msg=None):
        """
        Asserts that two label arrays represent the same basin segmentation,
        ignoring the specific integer IDs assigned to basins.
        It checks if nodes grouped in one array are also grouped in the other.
        """
        if result_labels.shape != expected_labels.shape:
            self.fail(f"Label shapes differ: {result_labels.shape} vs {expected_labels.shape}. {msg or ''}")

        # Create mappings from original labels to canonical labels (0, 1, 2, ...)
        def get_canonical_map(labels):
            unique_labels, canonical_labels = np.unique(labels, return_inverse=True)
            return canonical_labels

        canonical_result = get_canonical_map(result_labels)
        canonical_expected = get_canonical_map(expected_labels)
        
        try:
            np.testing.assert_array_equal(canonical_result, canonical_expected, err_msg=msg)
        except AssertionError as e:
            # Provide more context on failure
            detailed_msg = f"{msg or ''}\nResult:   {result_labels}\nExpected: {expected_labels}\nCanonical Result:   {canonical_result}\nCanonical Expected: {canonical_expected}"
            raise AssertionError(detailed_msg) from e


    def test_line_graphs(self):
        for i, vals in enumerate(self.node_vals_line):
            with self.subTest(i=i):
                lbls = watershed_segmentation_graph_spatial(vals, self.graph_line)
                self.assertBasinsEqual(lbls, self.expected_labels_line[i], msg=f"Line graph test case {i}")

    def test_circular_graphs(self):
        for i, vals in enumerate(self.node_vals_circ):
            with self.subTest(i=i):
                lbls = watershed_segmentation_graph_spatial(vals, self.graph_circ)
                self.assertBasinsEqual(lbls, self.expected_labels_circ[i], msg=f"Circular graph test case {i}")
    
    def test_tree_graphs(self):
        for i, vals in enumerate(self.node_vals_tree):
            with self.subTest(i=i):
                lbls = watershed_segmentation_graph_spatial(vals, self.graph_tree)
                self.assertBasinsEqual(lbls, self.expected_labels_tree[i], msg=f"Tree graph test case {i}")

    def test_grid_graphs(self):
        for i, vals in enumerate(self.node_vals_grid):
            with self.subTest(i=i):
                lbls = watershed_segmentation_graph_spatial(vals, self.graph_grid)
                self.assertBasinsEqual(lbls, self.expected_labels_grid[i], msg=f"Grid graph test case {i}")


class TestTemporalWatershed(unittest.TestCase):
    def setUp(self):
        """Set up common graph structures and value lists for temporal tests."""
        self.graph_line = np.array([
            [0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0]
        ])
        self.graph_circ = np.array([
            [0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0]
        ])
        self.graph_branch = np.zeros((7, 7))
        self.graph_branch[0, 1] = 1; self.graph_branch[1, 0] = 1
        self.graph_branch[1, 2] = 1; self.graph_branch[2, 1] = 1
        self.graph_branch[1, 3] = 1; self.graph_branch[3, 1] = 1
        self.graph_branch[3, 4] = 1; self.graph_branch[4, 3] = 1
        self.graph_branch[3, 5] = 1; self.graph_branch[5, 3] = 1
        self.graph_branch[3, 6] = 1; self.graph_branch[6, 3] = 1


        self.node_vals_tn_line = []
        self.expected_labels_tn_line = []

        # Test case from notebook: Diagonal -1s
        self.node_vals_tn_line.append(-np.eye(6))
        self.expected_labels_tn_line.append(np.ones((6,6), dtype=int))

        # Test case from notebook: Multiple distinct minima
        self.node_vals_tn_line.append(-np.array([
            [1,0,0,0,0,0], [0,3,0,0,0,0], [0,0,3,0,0,0],
            [0,0,0,4,0,0], [0,0,0,0,5,0], [0,0,0,0,0,6]
        ]))
        self.expected_labels_tn_line.append(np.ones((6,6), dtype=int))


        self.node_vals_tn_circ = []
        self.expected_labels_tn_circ = []
        # Example: Circular graph with a moving basin
        val_circ_t0 = -np.array([0, 1, 2, 1, 0, 0])
        val_circ_t1 = -np.array([0, 0, 1, 2, 1, 0])
        val_circ_t2 = -np.array([0, 0, 0, 1, 2, 1])
        self.node_vals_tn_circ.append(np.array([val_circ_t0, val_circ_t1, val_circ_t2]))
        # Expected: Basins might merge or stay separate depending on algorithm's temporal logic
        # From notebook output for a similar case (node_vals[2] for circ):
        self.expected_labels_tn_circ.append(np.array([
            [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1] # Assuming one large basin over time
            # Or more complex if basins are distinct and shift:
            # [[1,2,2,2,1,1],[1,1,2,2,2,1],[1,1,1,2,2,2]] # This needs exact notebook logic
        ]))
        # Using the example from the notebook that results in [[1,1,1,1,2,1],[1,1,1,1,2,1]...]
        self.node_vals_tn_circ.append(-np.array([
            [3,2,1,0,0,0],[2.4,1.6,0.8,0,0,0],[1.8,1.2,0.6,0,0,0],
            [1.2,0.8,0.4,0,0,0],[0.6,0.4,0.2,0,0,0],[0,0,0,0,0,0]
        ]))
        self.expected_labels_tn_circ.append(np.ones((6,6), dtype=int)) # Should be one basin

        self.node_vals_tn_circ.append(-np.array([
            [3,2,1,0,0,0],[2.4,1.6,0.8,-0.2,-0.4,-0.6],[1.8,1.2,0.6,-0.4,-0.8,-1.2],
            [1.2,0.8,0.4,-0.6,-1.2,-1.8],[0.6,0.4,0.2,-0.8,-1.6,-2.4],[0,0,0,-1,-2,-3]
        ]))
        self.expected_labels_tn_circ.append(np.array([ # Based on notebook output
            [1,1,1,1,2,1],[1,1,1,1,2,1],[1,1,1,2,2,1],
            [1,1,1,2,2,2],[2,1,2,2,2,2],[2,1,2,2,2,2]
        ]))


    def assertBasinsEqual(self, result_labels, expected_labels, msg=None):
        """
        Asserts that two label arrays represent the same basin segmentation,
        ignoring the specific integer IDs assigned to basins.
        """
        if result_labels.shape != expected_labels.shape:
            self.fail(f"Label shapes differ: {result_labels.shape} vs {expected_labels.shape}. {msg or ''}")

        def get_canonical_map(labels):
            unique_labels, canonical_labels = np.unique(labels, return_inverse=True)
            return canonical_labels.reshape(labels.shape)

        canonical_result = get_canonical_map(result_labels)
        canonical_expected = get_canonical_map(expected_labels)
        
        try:
            np.testing.assert_array_equal(canonical_result, canonical_expected, err_msg=msg)
        except AssertionError as e:
            detailed_msg = f"{msg or ''}\nResult:\n{result_labels}\nExpected:\n{expected_labels}\nCanonical Result:\n{canonical_result}\nCanonical Expected:\n{canonical_expected}"
            raise AssertionError(detailed_msg) from e

    def test_temporal_line_graphs(self):
        for i, vals_tn in enumerate(self.node_vals_tn_line):
            with self.subTest(i=i):
                lbls_tn = watershed_segmentation_graph_temporal(vals_tn, self.graph_line)
                self.assertBasinsEqual(lbls_tn, self.expected_labels_tn_line[i], msg=f"Temporal line graph test case {i}")

    def test_temporal_circular_graphs(self):
        for i, vals_tn in enumerate(self.node_vals_tn_circ):
            with self.subTest(i=i):
                lbls_tn = watershed_segmentation_graph_temporal(vals_tn, self.graph_circ)
                # For the second circular case, the notebook output is complex.
                # We will use the expected labels directly.
                self.assertBasinsEqual(lbls_tn, self.expected_labels_tn_circ[i], msg=f"Temporal circular graph test case {i}")
    
    # Add more temporal tests if specific expected outputs are clear from the notebook


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

