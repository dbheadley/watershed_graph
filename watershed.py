import numpy as np
import heapq
import collections
import pdb

def watershed_g(graph_matrix: np.ndarray, values_n: np.ndarray) -> np.ndarray:
    """
    Performs watershed segmentation on an undirected graph with static node values.
    Basins are seeded from regional minima plateaus: a plateau is a regional
    minimum if all its edge neighbors outside the plateau have
    strictly greater values. A plateau is a maximally connected region
    of nodes with the same elevation.

    Parameters
    ----------
    graph_matrix: numpy array (N,N)
        An N x N NumPy array representing the adjacency matrix of the graph.
        graph_matrix[i, j] == 1 if there's an edge between node i and node j,
        0 otherwise. The graph is assumed to be undirected.
    values_n: numpy array (N,)
        A 1D NumPy array of shape (N,), where N is the number of nodes.
        values_n[i] is the 'elevation' of node i.
    

    Returns
    -------
    labels_n: numpy array (N,)
        A 1D NumPy array of shape (N,), where each element labels_n[i]
        is an integer representing the basin ID assigned to node i.
        Basin IDs start from 1.
    """
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
    priority_queue = [] # Stores (value, entry_count, node_index)
    entry_count = 0
    current_basin_id = 0

    adj_list = [[] for _ in range(N)]
    rows, cols = np.where(graph_matrix == 1)
    for i, j in zip(rows, cols):
        adj_list[i].append(j)

    # --- Step 1: Marker Identification (Regional Minima Plateaus) ---
    sorted_node_indices = sorted(range(N), key=lambda i: (values_n[i], i))

    for n_start_plateau in sorted_node_indices:
        if labels_n[n_start_plateau] == 0: 
            current_plateau_value = values_n[n_start_plateau]
            
            plateau_nodes = [] 
            visited_for_this_plateau_search = set() 
            
            # BFS to find all spatially connected nodes with the same value as current_plateau_value
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
            
            # Check if this identified plateau is a true regional minimum
            is_regional_minimum_plateau = True
            for p_n in plateau_nodes: 
                for neighbor_n_check in adj_list[p_n]:
                    if neighbor_n_check not in visited_for_this_plateau_search: # If neighbor is outside the current plateau
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
    
    # --- Step 2: Flooding Phase ---
    while priority_queue:
        val_u, _, node_u = heapq.heappop(priority_queue) # val_u is values_n[node_u]
        basin_id_u = labels_n[node_u]
        if basin_id_u == 0: 
            continue

        for node_v_idx in adj_list[node_u]:
            if labels_n[node_v_idx] == 0: 
                labels_n[node_v_idx] = basin_id_u
                heapq.heappush(priority_queue, (values_n[node_v_idx], entry_count, node_v_idx))
                entry_count += 1
    return labels_n

def watershed_gt(graph_matrix: np.ndarray, values_tn: np.ndarray) -> np.ndarray:
    """
    Performs watershed segmentation on an undirected graph where node values
    change over time. The segmentation extends across time points.
    Basins are seeded from regional minima plateaus: a plateau is a regional
    minimum if all its edge-temporal neighbors outside the plateau have
    strictly greater values. A plateau is a maximally ST-connected region
    of points with the same elevation.

    Parameters
    ----------
    graph_matrix: numpy array (N,N)
        An N x N NumPy array representing the adjacency matrix of the graph.
        graph_matrix[i, j] == 1 if there's an edge between node i and node j,
        0 otherwise. The graph is assumed to be undirected and constant over time.
    values_tn: numpy array (T,N)
        A 2D NumPy array of shape (T, N), where T is the number of
        time points and N is the number of nodes.
        values_tn[t, i] is the 'elevation' of node i at time t.
    
    Returns
    -------
    labels_tn: numpy array (T,N)
        A 2D NumPy array of shape (T, N), where each element labels_tn[t, i]
        is an integer representing the basin ID assigned to node i at time t.
        Basin IDs start from 1.
    """
    if not isinstance(values_tn, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
        raise TypeError("Inputs 'values_tn' and 'graph_matrix' must be NumPy arrays.")
    if values_tn.ndim != 2:
        raise ValueError(f"Input 'values_tn' must be a 2D array (T, N), but got {values_tn.ndim} dimensions.")
    
    T, N = values_tn.shape
    if N == 0: 
        return np.array([[] for _ in range(T)], dtype=int) if T > 0 else np.array([], dtype=int)
    if T == 0: # N might be > 0 but T=0
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
        """Helper function to get unique spatio-temporal neighbors."""
        neighbors_set = set()
        # 1. Spatial neighbors at curr_t
        for neighbor_n in adj_list_local[curr_n]:
            neighbors_set.add((curr_t, neighbor_n))
        # 2. Self at curr_t +/- 1
        for dt_self in [-1, 1]:
            next_t_self = curr_t + dt_self
            if 0 <= next_t_self < T_max:
                neighbors_set.add((next_t_self, curr_n))
        # 3. Spatial neighbors of curr_n at curr_t +/- 1
        for dt_adj in [-1, 1]:
            adj_t = curr_t + dt_adj
            if 0 <= adj_t < T_max:
                for neighbor_n_adj_t in adj_list_local[curr_n]:
                    neighbors_set.add((adj_t, neighbor_n_adj_t))
        return list(neighbors_set)


    # --- Step 1: Marker Identification (Regional Minima Plateaus) ---
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
                    # Bounds check for nt, nn already handled by _get_st_neighbors for time,
                    # and adj_list for node indices.
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
    
    # --- Step 2: Flooding Phase ---
    while priority_queue:
        val_u, _, t_u, node_u = heapq.heappop(priority_queue) # val_u is values_tn[t_u, node_u]
        basin_id_u = labels_tn[t_u, node_u]
        if basin_id_u == 0: 
            continue

        # Spatial flooding at time t_u
        for node_v_spatial_idx in adj_list[node_u]:
            if labels_tn[t_u, node_v_spatial_idx] == 0: 
                labels_tn[t_u, node_v_spatial_idx] = basin_id_u
                heapq.heappush(priority_queue, (values_tn[t_u, node_v_spatial_idx], entry_count, t_u, node_v_spatial_idx))
                entry_count += 1
        
        # Temporal flooding/extension for node_u
        for dt in [-1, 1]:
            t_v_temporal = t_u + dt
            if 0 <= t_v_temporal < T:
                # Using val_u directly here (which is values_tn[t_u, node_u])
                if labels_tn[t_v_temporal, node_u] == 0 and \
                   values_tn[t_v_temporal, node_u] >= val_u:
                    labels_tn[t_v_temporal, node_u] = basin_id_u
                    heapq.heappush(priority_queue, (values_tn[t_v_temporal, node_u], entry_count, t_v_temporal, node_u))
                    entry_count += 1
    return labels_tn

# def watershed_segmentation_graph_temporal(values_tn: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
#     """
#     Performs watershed segmentation on an undirected graph where node values
#     change over time. The segmentation extends across time points.
#     Basins are seeded from regional minima plateaus: a plateau is a regional
#     minimum if all its spatio-temporal neighbors outside the plateau have
#     strictly greater values. A plateau is a maximally ST-connected region
#     of points with the same elevation.

#     Args:
#         values_tn: A 2D NumPy array of shape (T, N), where T is the number of
#                    time points and N is the number of nodes.
#                    values_tn[t, i] is the 'elevation' of node i at time t.
#         graph_matrix: An N x N NumPy array representing the adjacency matrix of the graph.
#                       graph_matrix[i, j] == 1 if there's an edge between node i and node j,
#                       0 otherwise. The graph is assumed to be undirected and constant over time.

#     Returns:
#         A 2D NumPy array of shape (T, N), where each element labels_tn[t, i]
#         is an integer representing the basin ID assigned to node i at time t.
#         Basin IDs start from 1.
#     """
#     if not isinstance(values_tn, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
#         raise TypeError("Inputs 'values_tn' and 'graph_matrix' must be NumPy arrays.")
#     if values_tn.ndim != 2:
#         raise ValueError(f"Input 'values_tn' must be a 2D array (T, N), but got {values_tn.ndim} dimensions.")
    
#     T, N = values_tn.shape
#     if N == 0: 
#         return np.array([[] for _ in range(T)], dtype=int) if T > 0 else np.array([], dtype=int)
#     if T == 0: # N might be > 0 but T=0
#         return np.array([], dtype=int).reshape(0,N)


#     if graph_matrix.ndim != 2 or graph_matrix.shape != (N, N):
#         raise ValueError(f"Input 'graph_matrix' must be an N x N ({N}x{N}) array, but got shape {graph_matrix.shape}.")

#     labels_tn = np.zeros((T, N), dtype=int)
#     priority_queue = []
#     entry_count = 0
#     current_basin_id = 0

#     adj_list = [[] for _ in range(N)]
#     rows, cols = np.where(graph_matrix == 1)
#     for i, j in zip(rows, cols):
#         adj_list[i].append(j)

#     # --- Step 1: Marker Identification (Regional Minima Plateaus) ---
#     sorted_indices_tn = sorted(
#         [(t, n) for t in range(T) for n in range(N)],
#         key=lambda x: (values_tn[x[0], x[1]], x[0], x[1])
#     )

#     for t_start_plateau, n_start_plateau in sorted_indices_tn:
#         if labels_tn[t_start_plateau, n_start_plateau] == 0: 
#             current_plateau_value = values_tn[t_start_plateau, n_start_plateau]
            
#             plateau_points = [] 
#             visited_for_this_plateau_search = set() 
            
#             # BFS to find all ST-connected points with the same value as current_plateau_value
#             # This defines the full extent of the current plateau.
#             bfs_plateau_q = collections.deque([(t_start_plateau, n_start_plateau)])
#             visited_for_this_plateau_search.add((t_start_plateau, n_start_plateau))

#             while bfs_plateau_q: 
#                 curr_t, curr_n = bfs_plateau_q.popleft()
#                 plateau_points.append((curr_t, curr_n)) 
                
#                 # Potential spatio-temporal neighbors to check for plateau expansion
#                 potential_st_neighbors_for_plateau = []
#                 # 1. Spatial neighbors at curr_t
#                 for neighbor_n in adj_list[curr_n]:
#                     potential_st_neighbors_for_plateau.append((curr_t, neighbor_n))
#                 # 2. Self at curr_t +/- 1
#                 for dt_self in [-1, 1]:
#                     next_t_self = curr_t + dt_self
#                     if 0 <= next_t_self < T:
#                         potential_st_neighbors_for_plateau.append((next_t_self, curr_n))
#                 # 3. Spatial neighbors of curr_n at curr_t +/- 1
#                 for dt_adj in [-1, 1]:
#                     adj_t = curr_t + dt_adj
#                     if 0 <= adj_t < T:
#                         for neighbor_n_adj_t in adj_list[curr_n]:
#                             potential_st_neighbors_for_plateau.append((adj_t, neighbor_n_adj_t))
                
#                 for nt, nn in potential_st_neighbors_for_plateau:
#                     # Check bounds for nt, nn (nn is always valid if from adj_list)
#                     if not (0 <= nt < T and 0 <= nn < N): continue # Should not happen for nn

#                     if (nt, nn) not in visited_for_this_plateau_search and \
#                        values_tn[nt, nn] == current_plateau_value:
#                         visited_for_this_plateau_search.add((nt, nn))
#                         bfs_plateau_q.append((nt, nn))
            
#             # Check if this identified plateau is a true regional minimum
#             is_regional_minimum_plateau = True
#             for p_t, p_n in plateau_points: # Iterate over all points found in the current plateau
#                 # Define ST-neighbors for the regional minimum check
#                 st_neighbors_for_min_check = []
#                 # 1. Spatial neighbors at p_t
#                 for neighbor_n in adj_list[p_n]:
#                     st_neighbors_for_min_check.append((p_t, neighbor_n))
#                 # 2. Self at p_t +/- 1
#                 for dt_self in [-1, 1]:
#                     next_t_self = p_t + dt_self
#                     if 0 <= next_t_self < T:
#                         st_neighbors_for_min_check.append((next_t_self, p_n))
#                 # 3. Spatial neighbors of p_n at p_t +/- 1
#                 for dt_adj in [-1, 1]:
#                     adj_t = p_t + dt_adj
#                     if 0 <= adj_t < T:
#                         for neighbor_n_adj_t in adj_list[p_n]:
#                             st_neighbors_for_min_check.append((adj_t, neighbor_n_adj_t))

#                 for nt_check, nn_check in st_neighbors_for_min_check:
#                     if not (0 <= nt_check < T and 0 <= nn_check < N): continue

#                     if (nt_check, nn_check) not in visited_for_this_plateau_search: # If neighbor is outside the current plateau
#                         if values_tn[nt_check, nn_check] <= current_plateau_value: 
#                             is_regional_minimum_plateau = False; break
#                 if not is_regional_minimum_plateau: break
            
#             if is_regional_minimum_plateau:
#                 current_basin_id += 1
#                 for p_t_label, p_n_label in plateau_points: # Use plateau_points which contains all members
#                     if labels_tn[p_t_label, p_n_label] == 0: 
#                         labels_tn[p_t_label, p_n_label] = current_basin_id
#                         heapq.heappush(priority_queue, (values_tn[p_t_label, p_n_label], entry_count, p_t_label, p_n_label))
#                         entry_count += 1
    
#     # --- Step 2: Flooding Phase ---
#     while priority_queue:
#         val_u, _, t_u, node_u = heapq.heappop(priority_queue)
#         basin_id_u = labels_tn[t_u, node_u]
#         if basin_id_u == 0: 
#             continue

#         # Spatial flooding at time t_u
#         for node_v_spatial_idx in adj_list[node_u]:
#             if labels_tn[t_u, node_v_spatial_idx] == 0: 
#                 labels_tn[t_u, node_v_spatial_idx] = basin_id_u
#                 heapq.heappush(priority_queue, (values_tn[t_u, node_v_spatial_idx], entry_count, t_u, node_v_spatial_idx))
#                 entry_count += 1
        
#         # Temporal flooding/extension for node_u
#         for dt in [-1, 1]:
#             t_v_temporal = t_u + dt
#             if 0 <= t_v_temporal < T:
#                 if labels_tn[t_v_temporal, node_u] == 0 and \
#                    values_tn[t_v_temporal, node_u] >= values_tn[t_u, node_u]:
#                     labels_tn[t_v_temporal, node_u] = basin_id_u
#                     heapq.heappush(priority_queue, (values_tn[t_v_temporal, node_u], entry_count, t_v_temporal, node_u))
#                     entry_count += 1
#     return labels_tn

# def watershed_segmentation_graph_temporal(values_tn: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
#     """
#     Performs watershed segmentation on an undirected graph where node values
#     change over time. The segmentation extends across time points.
#     Basins are seeded from regional minima plateaus: a plateau is a regional
#     minimum if all its spatio-temporal neighbors outside the plateau have
#     strictly greater values.

#     Args:
#         values_tn: A 2D NumPy array of shape (T, N), where T is the number of
#                    time points and N is the number of nodes.
#                    values_tn[t, i] is the 'elevation' of node i at time t.
#         graph_matrix: An N x N NumPy array representing the adjacency matrix of the graph.
#                       graph_matrix[i, j] == 1 if there's an edge between node i and node j,
#                       0 otherwise. The graph is assumed to be undirected and constant over time.

#     Returns:
#         A 2D NumPy array of shape (T, N), where each element labels_tn[t, i]
#         is an integer representing the basin ID assigned to node i at time t.
#         Basin IDs start from 1.
#     """
#     if not isinstance(values_tn, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
#         raise TypeError("Inputs 'values_tn' and 'graph_matrix' must be NumPy arrays.")
#     if values_tn.ndim != 2:
#         raise ValueError(f"Input 'values_tn' must be a 2D array (T, N), but got {values_tn.ndim} dimensions.")
    
#     T, N = values_tn.shape
#     if N == 0: 
#         return np.array([[] for _ in range(T)], dtype=int) if T > 0 else np.array([], dtype=int)
#     if T == 0: # N might be > 0 but T=0
#         return np.array([], dtype=int).reshape(0,N)


#     if graph_matrix.ndim != 2 or graph_matrix.shape != (N, N):
#         raise ValueError(f"Input 'graph_matrix' must be an N x N ({N}x{N}) array, but got shape {graph_matrix.shape}.")

#     labels_tn = np.zeros((T, N), dtype=int)
#     priority_queue = []
#     entry_count = 0
#     current_basin_id = 0

#     adj_list = [[] for _ in range(N)]
#     rows, cols = np.where(graph_matrix == 1)
#     for i, j in zip(rows, cols):
#         adj_list[i].append(j)

#     # --- Step 1: Marker Identification (Regional Minima Plateaus) ---
#     sorted_indices_tn = sorted(
#         [(t, n) for t in range(T) for n in range(N)],
#         key=lambda x: (values_tn[x[0], x[1]], x[0], x[1])
#     )

#     for t_start_plateau, n_start_plateau in sorted_indices_tn:
#         if labels_tn[t_start_plateau, n_start_plateau] == 0: 
#             current_plateau_value = values_tn[t_start_plateau, n_start_plateau]
            
#             plateau_points = [] 
#             visited_for_this_plateau_search = set() # Reset for each potential plateau start
            
#             # BFS to find all points belonging to this potential plateau
#             # temp_plateau_list acts as a queue for the BFS
#             temp_plateau_list = collections.deque([(t_start_plateau, n_start_plateau)])
#             visited_for_this_plateau_search.add((t_start_plateau, n_start_plateau))

#             while temp_plateau_list: # Use deque's boolean nature
#                 curr_t, curr_n = temp_plateau_list.popleft()
#                 plateau_points.append((curr_t, curr_n)) # Add to the list of points in this plateau
                
#                 # Explore spatial neighbors on plateau (same time, same value)
#                 for neighbor_n_spatial in adj_list[curr_n]:
#                     if (curr_t, neighbor_n_spatial) not in visited_for_this_plateau_search and \
#                        values_tn[curr_t, neighbor_n_spatial] == current_plateau_value:
#                         visited_for_this_plateau_search.add((curr_t, neighbor_n_spatial))
#                         temp_plateau_list.append((curr_t, neighbor_n_spatial))
                
#                 # Explore temporal neighbors on plateau (same node, same value)
#                 for dt in [-1, 1]:
#                     next_t = curr_t + dt
#                     if 0 <= next_t < T:
#                         if (next_t, curr_n) not in visited_for_this_plateau_search and \
#                            values_tn[next_t, curr_n] == current_plateau_value:
#                             visited_for_this_plateau_search.add((next_t, curr_n))
#                             temp_plateau_list.append((next_t, curr_n))
            
#             # Check if this identified plateau is a true regional minimum
#             # A plateau is a regional minimum if all its ST-neighbors *outside* the plateau
#             # have values *strictly greater* than the plateau's value.
#             is_regional_minimum_plateau = True
#             for p_t, p_n in plateau_points:
#                 # 1. Spatial neighbors at p_t
#                 for neighbor_n_spatial in adj_list[p_n]:
#                     if (p_t, neighbor_n_spatial) not in visited_for_this_plateau_search: 
#                         if values_tn[p_t, neighbor_n_spatial] <= current_plateau_value: # Changed from < to <=
#                             is_regional_minimum_plateau = False; break
#                 if not is_regional_minimum_plateau: break
                
#                 # 2. Self at p_t +/- 1 (Temporal neighbors of the point p_n)
#                 for dt_self in [-1, 1]:
#                     next_t_self = p_t + dt_self
#                     if 0 <= next_t_self < T:
#                         if (next_t_self, p_n) not in visited_for_this_plateau_search: 
#                             if values_tn[next_t_self, p_n] <= current_plateau_value: # Changed from < to <=
#                                 is_regional_minimum_plateau = False; break
#                 if not is_regional_minimum_plateau: break
                
#                 # 3. Spatial neighbors of p_n at p_t +/- 1
#                 for dt_adj in [-1, 1]:
#                     adj_t = p_t + dt_adj
#                     if 0 <= adj_t < T:
#                         for neighbor_n_spatial_adj_t in adj_list[p_n]:
#                              if (adj_t, neighbor_n_spatial_adj_t) not in visited_for_this_plateau_search:
#                                 if values_tn[adj_t, neighbor_n_spatial_adj_t] <= current_plateau_value: # Changed from < to <=
#                                     is_regional_minimum_plateau = False; break
#                         if not is_regional_minimum_plateau: break 
#                 if not is_regional_minimum_plateau: break 
            
#             if is_regional_minimum_plateau:
#                 current_basin_id += 1
#                 for p_t_label, p_n_label in plateau_points:
#                     if labels_tn[p_t_label, p_n_label] == 0: 
#                         labels_tn[p_t_label, p_n_label] = current_basin_id
#                         heapq.heappush(priority_queue, (values_tn[p_t_label, p_n_label], entry_count, p_t_label, p_n_label))
#                         entry_count += 1
    
#     # --- Step 2: Flooding Phase ---
#     while priority_queue:
#         val_u, _, t_u, node_u = heapq.heappop(priority_queue)
#         basin_id_u = labels_tn[t_u, node_u]
#         if basin_id_u == 0: 
#             continue

#         # Spatial flooding at time t_u
#         for node_v_spatial_idx in adj_list[node_u]:
#             if labels_tn[t_u, node_v_spatial_idx] == 0: 
#                 labels_tn[t_u, node_v_spatial_idx] = basin_id_u
#                 heapq.heappush(priority_queue, (values_tn[t_u, node_v_spatial_idx], entry_count, t_u, node_v_spatial_idx))
#                 entry_count += 1
        
#         # Temporal flooding/extension for node_u
#         for dt in [-1, 1]:
#             t_v_temporal = t_u + dt
#             if 0 <= t_v_temporal < T:
#                 if labels_tn[t_v_temporal, node_u] == 0 and \
#                    values_tn[t_v_temporal, node_u] >= values_tn[t_u, node_u]:
#                     labels_tn[t_v_temporal, node_u] = basin_id_u
#                     heapq.heappush(priority_queue, (values_tn[t_v_temporal, node_u], entry_count, t_v_temporal, node_u))
#                     entry_count += 1
#     return labels_tn

# def watershed_segmentation_graph_temporal(values_tn: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
#     """
#     Performs watershed segmentation on an undirected graph where node values
#     change over time. The segmentation extends across time points.
#     Basins are seeded from regional minima plateaus.

#     Args:
#         values_tn: A 2D NumPy array of shape (T, N), where T is the number of
#                    time points and N is the number of nodes.
#                    values_tn[t, i] is the 'elevation' of node i at time t.
#         graph_matrix: An N x N NumPy array representing the adjacency matrix of the graph.
#                       graph_matrix[i, j] == 1 if there's an edge between node i and node j,
#                       0 otherwise. The graph is assumed to be undirected and constant over time.

#     Returns:
#         A 2D NumPy array of shape (T, N), where each element labels_tn[t, i]
#         is an integer representing the basin ID assigned to node i at time t.
#         Basin IDs start from 1.
#     """
#     if not isinstance(values_tn, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
#         raise TypeError("Inputs 'values_tn' and 'graph_matrix' must be NumPy arrays.")
#     if values_tn.ndim != 2:
#         raise ValueError(f"Input 'values_tn' must be a 2D array (T, N), but got {values_tn.ndim} dimensions.")
    
#     T, N = values_tn.shape
#     if N == 0: 
#         return np.array([[] for _ in range(T)], dtype=int) if T > 0 else np.array([], dtype=int)
#     if T == 0: # N might be > 0 but T=0
#         return np.array([], dtype=int).reshape(0,N)


#     if graph_matrix.ndim != 2 or graph_matrix.shape != (N, N):
#         raise ValueError(f"Input 'graph_matrix' must be an N x N ({N}x{N}) array, but got shape {graph_matrix.shape}.")

#     labels_tn = np.zeros((T, N), dtype=int)
#     priority_queue = []
#     entry_count = 0
#     current_basin_id = 0

#     adj_list = [[] for _ in range(N)]
#     rows, cols = np.where(graph_matrix == 1)
#     for i, j in zip(rows, cols):
#         adj_list[i].append(j)

#     # --- Step 1: Marker Identification (Regional Minima Plateaus) ---
#     sorted_indices_tn = sorted(
#         [(t, n) for t in range(T) for n in range(N)],
#         key=lambda x: (values_tn[x[0], x[1]], x[0], x[1])
#     )

#     for t_start_plateau, n_start_plateau in sorted_indices_tn:
#         if labels_tn[t_start_plateau, n_start_plateau] == 0: 
#             current_plateau_value = values_tn[t_start_plateau, n_start_plateau]
            
#             plateau_points = [] # Stores (t,n) tuples of points in the current plateau
#             # BFS to find all points belonging to this potential plateau
#             plateau_q = collections.deque([(t_start_plateau, n_start_plateau)])
#             # visited_for_this_plateau_search stores points identified as part of the current plateau
#             visited_for_this_plateau_search = set([(t_start_plateau, n_start_plateau)])
            
#             head = 0
#             temp_plateau_list = [(t_start_plateau, n_start_plateau)] # Use list as deque for BFS

#             while head < len(temp_plateau_list):
#                 curr_t, curr_n = temp_plateau_list[head]
#                 head += 1
#                 plateau_points.append((curr_t, curr_n))
                
#                 # Explore spatial neighbors on plateau (same time, same value)
#                 for neighbor_n_spatial in adj_list[curr_n]:
#                     if (curr_t, neighbor_n_spatial) not in visited_for_this_plateau_search and \
#                        values_tn[curr_t, neighbor_n_spatial] == current_plateau_value:
#                         visited_for_this_plateau_search.add((curr_t, neighbor_n_spatial))
#                         temp_plateau_list.append((curr_t, neighbor_n_spatial))
                
#                 # Explore temporal neighbors on plateau (same node, same value)
#                 for dt in [-1, 1]:
#                     next_t = curr_t + dt
#                     if 0 <= next_t < T:
#                         if (next_t, curr_n) not in visited_for_this_plateau_search and \
#                            values_tn[next_t, curr_n] == current_plateau_value:
#                             visited_for_this_plateau_search.add((next_t, curr_n))
#                             temp_plateau_list.append((next_t, curr_n))
            
#             # Check if this identified plateau is a true regional minimum
#             is_regional_minimum_plateau = True
#             for p_t, p_n in plateau_points:
#                 # Check all ST-neighbors of (p_t, p_n)
#                 # 1. Spatial neighbors at p_t
#                 for neighbor_n_spatial in adj_list[p_n]:
#                     if (p_t, neighbor_n_spatial) not in visited_for_this_plateau_search: # If neighbor is outside the current plateau
#                         if values_tn[p_t, neighbor_n_spatial] < current_plateau_value:
#                             is_regional_minimum_plateau = False; break
#                 if not is_regional_minimum_plateau: break
                
#                 # 2. Self at p_t +/- 1 (Temporal neighbors of the point p_n)
#                 for dt_self in [-1, 1]:
#                     next_t_self = p_t + dt_self
#                     if 0 <= next_t_self < T:
#                         if (next_t_self, p_n) not in visited_for_this_plateau_search: 
#                             if values_tn[next_t_self, p_n] < current_plateau_value:
#                                 is_regional_minimum_plateau = False; break
#                 if not is_regional_minimum_plateau: break
                
#                 # 3. Spatial neighbors of p_n at p_t +/- 1
#                 for dt_adj in [-1, 1]:
#                     adj_t = p_t + dt_adj
#                     if 0 <= adj_t < T:
#                         for neighbor_n_spatial_adj_t in adj_list[p_n]:
#                              if (adj_t, neighbor_n_spatial_adj_t) not in visited_for_this_plateau_search:
#                                 if values_tn[adj_t, neighbor_n_spatial_adj_t] < current_plateau_value:
#                                     is_regional_minimum_plateau = False; break
#                         if not is_regional_minimum_plateau: break 
#                 if not is_regional_minimum_plateau: break 
            
#             if is_regional_minimum_plateau:
#                 current_basin_id += 1
#                 for p_t_label, p_n_label in plateau_points:
#                     # Only label if it hasn't been labeled by an even earlier (lower) regional minimum's plateau
#                     if labels_tn[p_t_label, p_n_label] == 0: 
#                         labels_tn[p_t_label, p_n_label] = current_basin_id
#                         heapq.heappush(priority_queue, (values_tn[p_t_label, p_n_label], entry_count, p_t_label, p_n_label))
#                         entry_count += 1
    
#     # --- Step 2: Flooding Phase ---
#     while priority_queue:
#         val_u, _, t_u, node_u = heapq.heappop(priority_queue)
#         basin_id_u = labels_tn[t_u, node_u]
#         if basin_id_u == 0: # Should ideally not happen if points are labeled before pushing
#             continue

#         # Spatial flooding at time t_u
#         for node_v_spatial_idx in adj_list[node_u]:
#             if labels_tn[t_u, node_v_spatial_idx] == 0: 
#                 labels_tn[t_u, node_v_spatial_idx] = basin_id_u
#                 heapq.heappush(priority_queue, (values_tn[t_u, node_v_spatial_idx], entry_count, t_u, node_v_spatial_idx))
#                 entry_count += 1
        
#         # Temporal flooding/extension for node_u
#         for dt in [-1, 1]:
#             t_v_temporal = t_u + dt
#             if 0 <= t_v_temporal < T:
#                 if labels_tn[t_v_temporal, node_u] == 0 and \
#                    values_tn[t_v_temporal, node_u] >= values_tn[t_u, node_u]:
#                     labels_tn[t_v_temporal, node_u] = basin_id_u
#                     heapq.heappush(priority_queue, (values_tn[t_v_temporal, node_u], entry_count, t_v_temporal, node_u))
#                     entry_count += 1
#     return labels_tn

# def watershed_segmentation_graph_temporal(values_tn: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
#     """
#     Performs watershed segmentation on an undirected graph where node values
#     change over time. The segmentation extends across time points.
#     A point (t,n) is a spatiotemporal local minimum if its value is less than
#     or equal to all its spatial neighbors at time t, its own value at t-1 and t+1,
#     and all its spatial neighbors at t-1 and t+1.

#     Args:
#         values_tn: A 2D NumPy array of shape (T, N), where T is the number of
#                    time points and N is the number of nodes.
#                    values_tn[t, i] is the 'elevation' of node i at time t.
#         graph_matrix: An N x N NumPy array representing the adjacency matrix of the graph.
#                       graph_matrix[i, j] == 1 if there's an edge between node i and node j,
#                       0 otherwise. The graph is assumed to be undirected and constant over time.

#     Returns:
#         A 2D NumPy array of shape (T, N), where each element labels_tn[t, i]
#         is an integer representing the basin ID assigned to node i at time t.
#         Basin IDs start from 1.
#     """
#     if not isinstance(values_tn, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
#         raise TypeError("Inputs 'values_tn' and 'graph_matrix' must be NumPy arrays.")
#     if values_tn.ndim != 2:
#         raise ValueError(f"Input 'values_tn' must be a 2D array (T, N), but got {values_tn.ndim} dimensions.")
    
#     T, N = values_tn.shape
#     if N == 0: # Handles T=0 or N=0
#         return np.array([[] for _ in range(T)], dtype=int) if T > 0 else np.array([], dtype=int)


#     if graph_matrix.ndim != 2 or graph_matrix.shape != (N, N):
#         raise ValueError(f"Input 'graph_matrix' must be an N x N ({N}x{N}) array, but got shape {graph_matrix.shape}.")

#     # labels_tn array: 0 means unlabelled, >0 means basin ID
#     labels_tn = np.zeros((T, N), dtype=int)
    
#     # Priority queue stores tuples: (value_at_tn, entry_count, time_idx, node_idx)
#     priority_queue = []
#     entry_count = 0
#     current_basin_id = 0

#     # Adjacency list for spatial connections (constant over time)
#     adj_list = [[] for _ in range(N)]
#     rows, cols = np.where(graph_matrix == 1)
#     for i, j in zip(rows, cols):
#         adj_list[i].append(j)

#     # Helper function to check for spatiotemporal local minima
#     def is_spatiotemporal_local_minimum(t_idx_local, node_idx_local, all_values_tn, adj_list_local, T_total):
#         node_val = all_values_tn[t_idx_local, node_idx_local]
        
#         # Check spatial neighbors at the current time t_idx_local
#         for neighbor_spatial_idx in adj_list_local[node_idx_local]:
#             if all_values_tn[t_idx_local, neighbor_spatial_idx] < node_val:
#                 return False 
        
#         # Check temporal neighbor of the current node at t_idx_local - 1 (past)
#         if t_idx_local > 0: 
#             if all_values_tn[t_idx_local - 1, node_idx_local] < node_val:
#                 return False
#             # Check spatial neighbors of current node at t_idx_local - 1
#             for neighbor_spatial_idx in adj_list_local[node_idx_local]:
#                 if all_values_tn[t_idx_local - 1, neighbor_spatial_idx] < node_val:
#                     return False
        
#         # Check temporal neighbor of the current node at t_idx_local + 1 (future)
#         if t_idx_local < T_total - 1: 
#             if all_values_tn[t_idx_local + 1, node_idx_local] < node_val:
#                 return False
#             # Check spatial neighbors of current node at t_idx_local + 1
#             for neighbor_spatial_idx in adj_list_local[node_idx_local]:
#                 if all_values_tn[t_idx_local + 1, neighbor_spatial_idx] < node_val:
#                     return False
        
#         return True # It's a spatiotemporal local minimum

#     # --- Step 1: Marker Identification (Spatio-Temporal Plateaus from Spatiotemporal Minima) ---
#     # Sort all (time, node) points by their value, then time, then node_idx for tie-breaking
#     sorted_indices_tn = sorted(
#         [(t, n) for t in range(T) for n in range(N)],
#         key=lambda x: (values_tn[x[0], x[1]], x[0], x[1])
#     )

#     for t_idx, node_idx in sorted_indices_tn:
#         if labels_tn[t_idx, node_idx] == 0:  # If not already labeled
#             # Use the updated spatiotemporal minimum check
#             if is_spatiotemporal_local_minimum(t_idx, node_idx, values_tn, adj_list, T):
#                 pdb.set_trace()
#                 # This (t_idx, node_idx) is a spatiotemporal local minimum
#                 current_basin_id += 1
                
#                 # BFS to find all connected points in this spatio-temporal plateau
#                 # A plateau consists of points with the *exact same value* as the seed.
#                 plateau_q = collections.deque()
#                 visited_in_plateau_bfs = set()

#                 # Seed the BFS for the plateau
#                 labels_tn[t_idx, node_idx] = current_basin_id
#                 heapq.heappush(priority_queue, (values_tn[t_idx, node_idx], entry_count, t_idx, node_idx))
#                 entry_count += 1
#                 plateau_q.append((t_idx, node_idx))
#                 visited_in_plateau_bfs.add((t_idx, node_idx))

#                 while plateau_q:
#                     curr_t, curr_n = plateau_q.popleft()
                    
#                     # Explore spatial neighbors on plateau (same time, same value)
#                     for neighbor_n_spatial in adj_list[curr_n]:
#                         if (curr_t, neighbor_n_spatial) not in visited_in_plateau_bfs and \
#                            values_tn[curr_t, neighbor_n_spatial] == values_tn[curr_t, curr_n] and \
#                            labels_tn[curr_t, neighbor_n_spatial] == 0: # Must be unlabeled by another basin seed
                            
#                             labels_tn[curr_t, neighbor_n_spatial] = current_basin_id
#                             heapq.heappush(priority_queue, (values_tn[curr_t, neighbor_n_spatial], entry_count, curr_t, neighbor_n_spatial))
#                             entry_count += 1
#                             plateau_q.append((curr_t, neighbor_n_spatial))
#                             visited_in_plateau_bfs.add((curr_t, neighbor_n_spatial))
                    
#                     # Explore temporal neighbors on plateau (same node, same value)
#                     for dt in [-1, 1]:
#                         next_t = curr_t + dt
#                         if 0 <= next_t < T:
#                             if (next_t, curr_n) not in visited_in_plateau_bfs and \
#                                values_tn[next_t, curr_n] == values_tn[curr_t, curr_n] and \
#                                labels_tn[next_t, curr_n] == 0: # Must be unlabeled by another basin seed

#                                 labels_tn[next_t, curr_n] = current_basin_id
#                                 heapq.heappush(priority_queue, (values_tn[next_t, curr_n], entry_count, next_t, curr_n))
#                                 entry_count += 1
#                                 plateau_q.append((next_t, curr_n))
#                                 visited_in_plateau_bfs.add((next_t, curr_n))
    
#     # --- Step 2: Flooding Phase ---
#     while priority_queue:
#         val_u, _, t_u, node_u = heapq.heappop(priority_queue)
        
#         basin_id_u = labels_tn[t_u, node_u]
#         if basin_id_u == 0: 
#             continue


#         # Spatial flooding at time t_u
#         for node_v_spatial_idx in adj_list[node_u]:
#             if labels_tn[t_u, node_v_spatial_idx] == 0:  # If neighbor is unlabelled
#                 labels_tn[t_u, node_v_spatial_idx] = basin_id_u
#                 heapq.heappush(priority_queue, (values_tn[t_u, node_v_spatial_idx], entry_count, t_u, node_v_spatial_idx))
#                 entry_count += 1
        
#         # Temporal flooding/extension for node_u
#         for dt in [-1, 1]:
#             t_v_temporal = t_u + dt
#             if 0 <= t_v_temporal < T:
#                 # Condition: unlabelled AND value at (t_v_temporal, node_u) is >= value at (t_u, node_u)
#                 # This allows flooding "uphill" or "same level" in time from a lower point.
#                 if labels_tn[t_v_temporal, node_u] == 0 and \
#                    values_tn[t_v_temporal, node_u] >= values_tn[t_u, node_u]:
                    
#                     labels_tn[t_v_temporal, node_u] = basin_id_u
#                     heapq.heappush(priority_queue, (values_tn[t_v_temporal, node_u], entry_count, t_v_temporal, node_u))
#                     entry_count += 1
#     return labels_tn

# def watershed_segmentation_graph_temporal(values_tn: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
#     """
#     Performs watershed segmentation on an undirected graph where node values
#     change over time. The segmentation extends across time points.
#     A point (t,n) is a spatiotemporal local minimum if its value is less than
#     or equal to all its spatial neighbors at time t, and less than or equal
#     to its own value at t-1 and t+1.

#     Args:
#         values_tn: A 2D NumPy array of shape (T, N), where T is the number of
#                    time points and N is the number of nodes.
#                    values_tn[t, i] is the 'elevation' of node i at time t.
#         graph_matrix: An N x N NumPy array representing the adjacency matrix of the graph.
#                       graph_matrix[i, j] == 1 if there's an edge between node i and node j,
#                       0 otherwise. The graph is assumed to be undirected and constant over time.

#     Returns:
#         A 2D NumPy array of shape (T, N), where each element labels_tn[t, i]
#         is an integer representing the basin ID assigned to node i at time t.
#         Basin IDs start from 1.
#     """
#     if not isinstance(values_tn, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
#         raise TypeError("Inputs 'values_tn' and 'graph_matrix' must be NumPy arrays.")
#     if values_tn.ndim != 2:
#         raise ValueError(f"Input 'values_tn' must be a 2D array (T, N), but got {values_tn.ndim} dimensions.")
    
#     T, N = values_tn.shape
#     if N == 0: # Handles T=0 or N=0
#         return np.array([[] for _ in range(T)], dtype=int) if T > 0 else np.array([], dtype=int)


#     if graph_matrix.ndim != 2 or graph_matrix.shape != (N, N):
#         raise ValueError(f"Input 'graph_matrix' must be an N x N ({N}x{N}) array, but got shape {graph_matrix.shape}.")

#     # labels_tn array: 0 means unlabelled, >0 means basin ID
#     labels_tn = np.zeros((T, N), dtype=int)
    
#     # Priority queue stores tuples: (value_at_tn, entry_count, time_idx, node_idx)
#     priority_queue = []
#     entry_count = 0
#     current_basin_id = 0

#     # Adjacency list for spatial connections (constant over time)
#     adj_list = [[] for _ in range(N)]
#     rows, cols = np.where(graph_matrix == 1)
#     for i, j in zip(rows, cols):
#         adj_list[i].append(j)

#     # Helper function to check for spatiotemporal local minima
#     def is_spatiotemporal_local_minimum(t_idx_local, node_idx_local, all_values_tn, adj_list_local, T_total):
#         node_val = all_values_tn[t_idx_local, node_idx_local]
        
#         # Check spatial neighbors at the same time t_idx_local
#         for neighbor_spatial_idx in adj_list_local[node_idx_local]:
#             if all_values_tn[t_idx_local, neighbor_spatial_idx] < node_val:
#                 return False # Not a minimum if any spatial neighbor is strictly smaller
        
#         # Check temporal neighbor at t_idx_local - 1 (past)
#         if t_idx_local > 0: # If a past time step exists
#             if all_values_tn[t_idx_local - 1, node_idx_local] < node_val:
#                 return False # Not a minimum if past self is strictly smaller
        
#         # Check temporal neighbor at t_idx_local + 1 (future)
#         if t_idx_local < T_total - 1: # If a future time step exists
#             if all_values_tn[t_idx_local + 1, node_idx_local] < node_val:
#                 return False # Not a minimum if future self is strictly smaller
        
#         return True # It's a spatiotemporal local minimum

#     # --- Step 1: Marker Identification (Spatio-Temporal Plateaus from Spatiotemporal Minima) ---
#     # Sort all (time, node) points by their value, then time, then node_idx for tie-breaking
#     sorted_indices_tn = sorted(
#         [(t, n) for t in range(T) for n in range(N)],
#         key=lambda x: (values_tn[x[0], x[1]], x[0], x[1])
#     )

#     for t_idx, node_idx in sorted_indices_tn:
#         if labels_tn[t_idx, node_idx] == 0:  # If not already labeled
#             # Use the updated spatiotemporal minimum check
#             if is_spatiotemporal_local_minimum(t_idx, node_idx, values_tn, adj_list, T):
#                 # This (t_idx, node_idx) is a spatiotemporal local minimum
#                 current_basin_id += 1
                
#                 # BFS to find all connected points in this spatio-temporal plateau
#                 # A plateau consists of points with the *exact same value* as the seed.
#                 plateau_q = collections.deque()
#                 visited_in_plateau_bfs = set()

#                 # Seed the BFS for the plateau
#                 labels_tn[t_idx, node_idx] = current_basin_id
#                 heapq.heappush(priority_queue, (values_tn[t_idx, node_idx], entry_count, t_idx, node_idx))
#                 entry_count += 1
#                 plateau_q.append((t_idx, node_idx))
#                 visited_in_plateau_bfs.add((t_idx, node_idx))

#                 while plateau_q:
#                     curr_t, curr_n = plateau_q.popleft()
                    
#                     # Explore spatial neighbors on plateau (same time, same value)
#                     for neighbor_n_spatial in adj_list[curr_n]:
#                         if (curr_t, neighbor_n_spatial) not in visited_in_plateau_bfs and \
#                            values_tn[curr_t, neighbor_n_spatial] == values_tn[curr_t, curr_n] and \
#                            labels_tn[curr_t, neighbor_n_spatial] == 0: # Must be unlabeled by another basin seed
                            
#                             labels_tn[curr_t, neighbor_n_spatial] = current_basin_id
#                             heapq.heappush(priority_queue, (values_tn[curr_t, neighbor_n_spatial], entry_count, curr_t, neighbor_n_spatial))
#                             entry_count += 1
#                             plateau_q.append((curr_t, neighbor_n_spatial))
#                             visited_in_plateau_bfs.add((curr_t, neighbor_n_spatial))
                    
#                     # Explore temporal neighbors on plateau (same node, same value)
#                     for dt in [-1, 1]:
#                         next_t = curr_t + dt
#                         if 0 <= next_t < T:
#                             if (next_t, curr_n) not in visited_in_plateau_bfs and \
#                                values_tn[next_t, curr_n] == values_tn[curr_t, curr_n] and \
#                                labels_tn[next_t, curr_n] == 0: # Must be unlabeled by another basin seed

#                                 labels_tn[next_t, curr_n] = current_basin_id
#                                 heapq.heappush(priority_queue, (values_tn[next_t, curr_n], entry_count, next_t, curr_n))
#                                 entry_count += 1
#                                 plateau_q.append((next_t, curr_n))
#                                 visited_in_plateau_bfs.add((next_t, curr_n))
    
#     # --- Step 2: Flooding Phase ---
#     while priority_queue:
#         val_u, _, t_u, node_u = heapq.heappop(priority_queue)
        
#         basin_id_u = labels_tn[t_u, node_u]
#         if basin_id_u == 0: 
#             # This might happen if a point was part of a plateau of a *later* (higher value) minimum
#             # but was already flooded by an *earlier* (lower value) minimum.
#             # Or if a point was added to PQ but then got labeled by another branch of flooding from the same basin.
#             # If it has a valid label from an earlier flooding, that's fine.
#             # If it's truly 0, it means it wasn't properly seeded or part of a plateau, which is unlikely here.
#             # For safety, we can check if it has a label, and if that label matches what we expect.
#             # However, the core logic is that if it has *any* non-zero label, it's processed.
#             # If it's 0, it means it hasn't been reached by any seed's plateau or flood yet,
#             # which implies it should have been a seed itself if it was a minimum, or it's higher up.
#             # Given the PQ processes lowest values first, a 0 label here is odd unless it was never a seed.
#             # The current logic: if basin_id_u is 0, this point was never part of an initial seed plateau.
#             # This implies it should be flooded by something.
#             # However, points are added to PQ *after* being labeled in the seeding/plateau phase.
#             # So basin_id_u should always be > 0 here.
#             # The original `if basin_id_u == 0: continue` is a safeguard.
#             continue


#         # Spatial flooding at time t_u
#         for node_v_spatial_idx in adj_list[node_u]:
#             if labels_tn[t_u, node_v_spatial_idx] == 0:  # If neighbor is unlabelled
#                 labels_tn[t_u, node_v_spatial_idx] = basin_id_u
#                 heapq.heappush(priority_queue, (values_tn[t_u, node_v_spatial_idx], entry_count, t_u, node_v_spatial_idx))
#                 entry_count += 1
        
#         # Temporal flooding/extension for node_u
#         for dt in [-1, 1]:
#             t_v_temporal = t_u + dt
#             if 0 <= t_v_temporal < T:
#                 # Condition: unlabelled AND value at (t_v_temporal, node_u) is >= value at (t_u, node_u)
#                 # This allows flooding "uphill" or "same level" in time from a lower point.
#                 if labels_tn[t_v_temporal, node_u] == 0 and \
#                    values_tn[t_v_temporal, node_u] >= values_tn[t_u, node_u]:
                    
#                     labels_tn[t_v_temporal, node_u] = basin_id_u
#                     heapq.heappush(priority_queue, (values_tn[t_v_temporal, node_u], entry_count, t_v_temporal, node_u))
#                     entry_count += 1
#     return labels_tn

# def watershed_segmentation_graph_temporal(values_tn: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
#     """
#     Performs watershed segmentation on an undirected graph where node values
#     change over time. The segmentation extends across time points.

#     Args:
#         values_tn: A 2D NumPy array of shape (T, N), where T is the number of
#                    time points and N is the number of nodes.
#                    values_tn[t, i] is the 'elevation' of node i at time t.
#         graph_matrix: An N x N NumPy array representing the adjacency matrix of the graph.
#                       graph_matrix[i, j] == 1 if there's an edge between node i and node j,
#                       0 otherwise. The graph is assumed to be undirected and constant over time.

#     Returns:
#         A 2D NumPy array of shape (T, N), where each element labels_tn[t, i]
#         is an integer representing the basin ID assigned to node i at time t.
#         Basin IDs start from 1.
#     """
#     if not isinstance(values_tn, np.ndarray) or not isinstance(graph_matrix, np.ndarray):
#         raise TypeError("Inputs 'values_tn' and 'graph_matrix' must be NumPy arrays.")
#     if values_tn.ndim != 2:
#         raise ValueError(f"Input 'values_tn' must be a 2D array (T, N), but got {values_tn.ndim} dimensions.")
    
#     T, N = values_tn.shape
#     if N == 0: # Handles T=0 or N=0
#         return np.array([[] for _ in range(T)], dtype=int) if T > 0 else np.array([], dtype=int)


#     if graph_matrix.ndim != 2 or graph_matrix.shape != (N, N):
#         raise ValueError(f"Input 'graph_matrix' must be an N x N ({N}x{N}) array, but got shape {graph_matrix.shape}.")

#     # labels_tn array: 0 means unlabelled, >0 means basin ID
#     labels_tn = np.zeros((T, N), dtype=int)
    
#     # Priority queue stores tuples: (value_at_tn, entry_count, time_idx, node_idx)
#     priority_queue = []
#     entry_count = 0
#     current_basin_id = 0

#     # Adjacency list for spatial connections (constant over time)
#     adj_list = [[] for _ in range(N)]
#     rows, cols = np.where(graph_matrix == 1)
#     for i, j in zip(rows, cols):
#         adj_list[i].append(j)

#     # Helper function to check for spatial local minima at a given time
#     def is_spatial_local_minimum(t_idx_local, node_idx_local, all_values_tn, adj_list_local):
#         node_val = all_values_tn[t_idx_local, node_idx_local]
#         if not adj_list_local[node_idx_local]: # Node has no spatial neighbors
#             return True
#         for neighbor_spatial_idx in adj_list_local[node_idx_local]:
#             if all_values_tn[t_idx_local, neighbor_spatial_idx] < node_val:
#                 return False
#         return True

#     # --- Step 1: Marker Identification (Spatio-Temporal Plateaus from Spatial Minima) ---
#     # Sort all (time, node) points by their value, then time, then node_idx for tie-breaking
#     sorted_indices_tn = sorted(
#         [(t, n) for t in range(T) for n in range(N)],
#         key=lambda x: (values_tn[x[0], x[1]], x[0], x[1])
#     )

#     for t_idx, node_idx in sorted_indices_tn:
#         if labels_tn[t_idx, node_idx] == 0:  # If not already labeled
#             if is_spatial_local_minimum(t_idx, node_idx, values_tn, adj_list):
#                 # This (t_idx, node_idx) is a spatial local minimum (or start of one)
#                 # and hasn't been claimed by an earlier, lower seed's plateau.
#                 current_basin_id += 1
                
#                 # BFS to find all connected points in this spatio-temporal plateau
#                 plateau_q = collections.deque()
#                 visited_in_plateau_bfs = set()

#                 # Seed the BFS for the plateau
#                 labels_tn[t_idx, node_idx] = current_basin_id
#                 heapq.heappush(priority_queue, (values_tn[t_idx, node_idx], entry_count, t_idx, node_idx))
#                 entry_count += 1
#                 plateau_q.append((t_idx, node_idx))
#                 visited_in_plateau_bfs.add((t_idx, node_idx))

#                 while plateau_q:
#                     curr_t, curr_n = plateau_q.popleft()
                    
#                     # Explore spatial neighbors on plateau (same time, same value)
#                     for neighbor_n_spatial in adj_list[curr_n]:
#                         if (curr_t, neighbor_n_spatial) not in visited_in_plateau_bfs and \
#                            values_tn[curr_t, neighbor_n_spatial] == values_tn[curr_t, curr_n] and \
#                            labels_tn[curr_t, neighbor_n_spatial] == 0:
                            
#                             labels_tn[curr_t, neighbor_n_spatial] = current_basin_id
#                             heapq.heappush(priority_queue, (values_tn[curr_t, neighbor_n_spatial], entry_count, curr_t, neighbor_n_spatial))
#                             entry_count += 1
#                             plateau_q.append((curr_t, neighbor_n_spatial))
#                             visited_in_plateau_bfs.add((curr_t, neighbor_n_spatial))
                    
#                     # Explore temporal neighbors on plateau (same node, same value)
#                     for dt in [-1, 1]:
#                         next_t = curr_t + dt
#                         if 0 <= next_t < T:
#                             if (next_t, curr_n) not in visited_in_plateau_bfs and \
#                                values_tn[next_t, curr_n] == values_tn[curr_t, curr_n] and \
#                                labels_tn[next_t, curr_n] == 0:

#                                 labels_tn[next_t, curr_n] = current_basin_id
#                                 heapq.heappush(priority_queue, (values_tn[next_t, curr_n], entry_count, next_t, curr_n))
#                                 entry_count += 1
#                                 plateau_q.append((next_t, curr_n))
#                                 visited_in_plateau_bfs.add((next_t, curr_n))
    
#     # --- Step 2: Flooding Phase ---
#     while priority_queue:
#         val_u, _, t_u, node_u = heapq.heappop(priority_queue)
        
#         basin_id_u = labels_tn[t_u, node_u]
#         if basin_id_u == 0: # Should not happen if seeding is correct
#             continue

#         # Spatial flooding at time t_u
#         for node_v_spatial_idx in adj_list[node_u]:
#             if labels_tn[t_u, node_v_spatial_idx] == 0:  # If neighbor is unlabelled
#                 labels_tn[t_u, node_v_spatial_idx] = basin_id_u
#                 heapq.heappush(priority_queue, (values_tn[t_u, node_v_spatial_idx], entry_count, t_u, node_v_spatial_idx))
#                 entry_count += 1
        
#         # Temporal flooding/extension for node_u
#         for dt in [-1, 1]:
#             t_v_temporal = t_u + dt
#             if 0 <= t_v_temporal < T:
#                 # Condition: unlabelled AND value at (t_v_temporal, node_u) is >= value at (t_u, node_u)
#                 if labels_tn[t_v_temporal, node_u] == 0 and \
#                    values_tn[t_v_temporal, node_u] >= values_tn[t_u, node_u]:
                    
#                     labels_tn[t_v_temporal, node_u] = basin_id_u
#                     heapq.heappush(priority_queue, (values_tn[t_v_temporal, node_u], entry_count, t_v_temporal, node_u))
#                     entry_count += 1
#     return labels_tn