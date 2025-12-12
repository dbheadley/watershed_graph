import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from watershed import watershed_g, watershed_gt

def plot_g(graph_matrix: np.ndarray, node_values: np.ndarray, node_labels: np.ndarray):
    """
    Visualizes the result of a spatial watershed segmentation on a graph.

    Node fill color is determined by the node's value (elevation).
    Node border color is determined by the node's basin label.

    Args:
        graph_matrix: The N x N adjacency matrix of the graph.
        node_values: A 1D array of N elevation values for the nodes.
        node_labels: A 1D array of N integer basin labels for the nodes.
    """
    if graph_matrix.shape[0] != len(node_values) or len(node_values) != len(node_labels):
        raise ValueError("Graph, values, and labels must have the same number of nodes.")
    
    G = nx.from_numpy_array(graph_matrix)
    
    # Use a deterministic layout for consistent visualization
    pos = nx.kamada_kawai_layout(G)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- Node Fill Color (from values) ---
    value_cmap = cm.get_cmap('viridis')
    value_norm = mcolors.Normalize(vmin=node_values.min(), vmax=node_values.max())
    node_fill_colors = [value_cmap(value_norm(val)) for val in node_values]

    # --- Node Border Color (from labels) ---
    unique_labels = sorted(np.unique(node_labels))
    label_cmap = cm.get_cmap('tab10', len(unique_labels))
    label_to_color_map = {label: label_cmap(i) for i, label in enumerate(unique_labels)}
    node_border_colors = [label_to_color_map[label] for label in node_labels]

    # --- Draw the graph ---
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6)
    
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, 
        node_color=node_fill_colors,
        edgecolors=node_border_colors,
        linewidths=2.5,
        node_size=800
    )
    
    # Add text labels (basin IDs) on nodes
    label_text = {i: str(label) for i, label in enumerate(node_labels)}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=label_text, font_color='white', font_weight='bold')
    
    # --- Add a colorbar for the node values ---
    sm = plt.cm.ScalarMappable(cmap=value_cmap, norm=value_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Node Value (Elevation)', weight='bold')

    ax.set_title("Spatial Watershed Segmentation Result", fontsize=16, weight='bold')
    plt.show()


def plot_gt(graph_matrix: np.ndarray, values_tn: np.ndarray, labels_tn: np.ndarray, filename=None):
    """
    Visualizes the result of a temporal watershed segmentation as an animation.

    Args:
        graph_matrix: The N x N adjacency matrix of the graph.
        values_tn: A T x N array of node values over time.
        labels_tn: A T x N array of node labels over time.
        filename (str, optional): Path to save the animation (e.g., 'animation.gif'). 
                                  If None, displays in the environment.
    """
    T, N = values_tn.shape
    if graph_matrix.shape[0] != N or labels_tn.shape != (T, N):
        raise ValueError("Graph, values, and labels dimensions are inconsistent.")

    G = nx.from_numpy_array(graph_matrix)
    pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Setup consistent colormaps across all timesteps ---
    # Value colormap
    value_cmap = cm.get_cmap('viridis')
    value_norm = mcolors.Normalize(vmin=values_tn.min(), vmax=values_tn.max())

    # Label colormap
    unique_labels = sorted(np.unique(labels_tn))
    label_cmap = cm.get_cmap('tab10', len(unique_labels) if unique_labels.size > 0 else 1)
    label_to_color_map = {label: label_cmap(i) for i, label in enumerate(unique_labels)}
    
    # Initial drawing of the graph structure
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6)
    
    # Draw initial nodes that will be updated in the animation
    initial_fill_colors = [value_cmap(value_norm(val)) for val in values_tn[0, :]]
    initial_border_colors = [label_to_color_map.get(label, 'black') for label in labels_tn[0, :]]

    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=initial_fill_colors,
        edgecolors=initial_border_colors,
        linewidths=2.5,
        node_size=800
    )
    
    # Add a colorbar for the node values
    sm = plt.cm.ScalarMappable(cmap=value_cmap, norm=value_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Node Value (Elevation)', weight='bold')

    # Add labels (node indices) which will not change
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_weight='bold')

    def update(t):
        """Update function for the animation for timestep t."""
        node_fill_colors = [value_cmap(value_norm(val)) for val in values_tn[t, :]]
        node_border_colors = [label_to_color_map.get(label, 'black') for label in labels_tn[t, :]]
        
        # Update node properties efficiently
        nodes.set_facecolor(node_fill_colors)
        nodes.set_edgecolor(node_border_colors)
        
        ax.set_title(f"Temporal Watershed Segmentation (Time Step: {t})", fontsize=16, weight='bold')
        return nodes,

    anim = FuncAnimation(fig, update, frames=T, interval=500, blit=True)

    if filename:
        print(f"Saving animation to {filename}...")
        # Note: Saving may require ffmpeg or imagemagick to be installed
        anim.save(filename, writer='imagemagick', fps=2)
        print("Save complete.")
    else:
        # Display in a Jupyter-like environment
        plt.close(fig) # Prevent static figure from displaying below the animation
        return HTML(anim.to_html5_video())


if __name__ == '__main__':
    # --- Example Data Setup ---
    
    # 1. Spatial Line Graph Example
    graph_line = np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0]
    ])
    node_vals_line = -np.array([1, 2, 1, 0, 1, 2])
    labels_line = watershed_g(node_vals_line, graph_line)
    
    print("--- Visualizing Spatial Segmentation (Line Graph) ---")
    plot_g(graph_line, node_vals_line, labels_line)

    # 2. Temporal Line Graph Example
    T_steps, N_nodes = 6, 6
    # A basin that moves from left to right over time
    values_temporal_line = np.full((T_steps, N_nodes), 5.0)
    for t in range(T_steps):
        values_temporal_line[t, t] = -2.0 # Minimum value
        if t > 0:
            values_temporal_line[t, t-1] = 2.0
        if t < N_nodes - 1:
            values_temporal_line[t, t+1] = 2.0
    
    labels_temporal_line = watershed_gt(values_temporal_line, graph_line)

    print("\n--- Generating Temporal Segmentation Animation (Line Graph) ---")
    # This will return an animatable object in environments like Jupyter
    animation = plot_gt(graph_line, values_temporal_line, labels_temporal_line)
    
    # If in a script and you want to save:
    # visualize_temporal_segmentation(graph_line, values_temporal_line, labels_temporal_line, filename='temporal_line.gif')

    # To display the animation if the return object is generated
    if animation:
        from IPython.display import display
        display(animation)

