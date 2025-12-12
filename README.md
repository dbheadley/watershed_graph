# Graph-Based Watershed Segmentation

This repository contains Python implementations of the watershed segmentation algorithm adapted for graph-based data, supporting both static (spatial) and time-varying (spatio-temporal) node values.

## Introduction to the Watershed Algorithm

The watershed algorithm is a classic segmentation method originally developed for image processing. It treats a dataset as a topographical landscape where numerical values represent elevation. The core idea is to identify distinct "catchment basins" by flooding the landscape from its local minima. The lines where water from different basins would meet are called "watershed lines" or "dams," which form the boundaries between segments.

This implementation applies the same concept to an undirected graph where each node has an associated "elevation" value.

## Core Concepts

-   **Graph**: The underlying structure is an undirected graph, defined by an N x N adjacency matrix, where N is the number of nodes.
-   **Elevation**: Each node has a numerical value that serves as its elevation.
-   **Local Minima / Basin**: A node (or a connected plateau of nodes with the same elevation) whose value is less than or equal to all of its neighbors. These serve as the starting points, or "seeds," for the basins.
-   **Flooding**: Starting from the identified local minima, labels (basin IDs) are propagated to adjacent, unlabeled nodes with higher or equal elevation, simulating the process of water rising and filling a basin.
-   **Watershed Lines (Dams)**: These are the implicit boundaries where the "flooding" from two different basins meets. The algorithm assigns nodes to the first basin that reaches them, naturally forming these dividing lines.

---

## Algorithms

Two primary functions are provided to handle different types of data.

### 1. Spatial Watershed Algorithm (`watershed_g`)

This function performs watershed segmentation on a static graph where node elevations do not change.

**Algorithm Steps:**

1.  **Identify Regional Minima Plateaus**:
    * All nodes are sorted by their elevation value, from lowest to highest.
    * The algorithm iterates through the sorted nodes. If it finds an unlabeled node, it performs a Breadth-First Search (BFS) to discover the entire connected region of nodes that share the *exact same elevation*. This region is called a "plateau."
    * The key step is to verify if this plateau is a true "regional minimum." It qualifies as a minimum only if **all** of its neighboring nodes *outside* the plateau have a **strictly greater** elevation. This prevents plateaus on a slope from being incorrectly identified as new basins.

2.  **Seed and Flood**:
    * If a plateau is confirmed as a regional minimum, it is assigned a new, unique basin ID. All nodes within this plateau are added to a queue ordered from lowest to highest value.
    * The algorithm then iteratively processes the priority queue. It takes the node with the lowest elevation and "floods" its basin ID to any of its adjacent, unlabeled neighbors.
    * These newly labeled neighbors are then added to the queue to continue the process.
    * The flooding naturally stops when it encounters nodes that have already been labeled by another basin.

### 2. Spatio-temporal Watershed Algorithm (`watershed_gt`)

This function extends the watershed concept to data where node values evolve over time. The graph structure remains constant, but the input values are a T x N matrix, where T is the number of time points.

**Key Differences and Algorithm Steps:**

1.  **Spatio-temporal Neighborhood**: The concept of a "neighbor" is expanded. A point `(t, n)` (node `n` at time `t`) has neighbors:
    * **Spatial**: `(t, n_neighbor)`
    * **Temporal**: `(t-1, n)` and `(t+1, n)`
    * **Spatio-temporal**: `(t-1, n_neighbor)` and `(t+1, n_neighbor)`

2.  **Identify Spatio-temporal Regional Minima**:
    * All points `(t, n)` are sorted globally by their elevation value.
    * Similar to the spatial version, the algorithm discovers connected plateaus of points with the exact same elevation. However, the BFS for this discovery explores the full spatio-temporal neighborhood defined above.
    * A plateau is confirmed as a regional minimum only if all its spatio-temporal neighbors *outside* the plateau have a **strictly greater** elevation.

3.  **Seed and Flood (Spatial and Temporal)**:
    * Once a regional minimum plateau is seeded, the flooding process also occurs in two ways from any point `(t_u, n_u)` in the priority queue:
        * **Spatial Flooding**: The basin ID propagates to unlabeled spatial neighbors at the same time step, `(t_u, n_v)`.
        * **Temporal Flooding**: The basin ID can also propagate forward or backward in time to the same node, `(t_v, n_u)`, if that point is unlabeled and its elevation is **greater than or equal to** the current point's elevation. This allows basins to persist or grow over time.

---

## Usage

The functions are designed to be used with NumPy arrays.

```python
import numpy as np
# from watershed_algorithms import watershed_segmentation_graph_spatial, watershed_segmentation_graph_temporal

# --- Spatial Example ---
# Adjacency matrix for a 6-node line graph
graph_line = np.array([
    [0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0]
])
# Elevation values for the 6 nodes
node_values = -np.array([1, 2, 3, 2, 1, 0])

# Perform segmentation
spatial_labels = watershed_segmentation_graph_spatial(node_values, graph_line)
# Result: A 1D array of basin labels for each node.

# --- Spatio-temporal Example ---
# Values for 6 nodes over 6 time steps
node_values_tn = -np.eye(6)

# Perform segmentation
temporal_labels = watershed_segmentation_graph_temporal(node_values_tn, graph_line)
# Result: A 2D (6x6) array of basin labels.

##Testing

A comprehensive suite of unit tests is provided in test_watershed_algorithms.py. These tests use the unittest framework to validate the behavior of both algorithms against a variety of graph structures (line, circular, tree, grid) and value configurations, including the specific edge cases that were identified during development.</markdown>
