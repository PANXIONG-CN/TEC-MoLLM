import numpy as np
import logging
import torch
from scipy.sparse import coo_matrix, diags
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import os

# Add an import to the data loader we just created
from src.data.data_loader import load_and_split_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_coordinates_from_data(file_paths: list):
    """
    Loads data and extracts the latitude and longitude arrays.
    """
    data = load_and_split_data(file_paths)
    if not data:
        logging.error("Failed to load data to get coordinates.")
        return None, None

    # Coordinates are static, so we can take them from any split (e.g., train)
    if "latitude" in data["train"] and "longitude" in data["train"]:
        lat = data["train"]["latitude"]
        lon = data["train"]["longitude"]
        logging.info(f"Successfully loaded coordinates. Latitude shape: {lat.shape}, Longitude shape: {lon.shape}")
        return lat, lon
    else:
        logging.error("Latitude or Longitude not found in the loaded data.")
        return None, None


def calculate_haversine_distance_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculates the pairwise Haversine distance matrix for a grid of coordinates.

    Args:
        lat (np.ndarray): 1D array of latitudes.
        lon (np.ndarray): 1D array of longitudes.

    Returns:
        np.ndarray: A 2D matrix of pairwise distances in kilometers.
    """
    # Create a meshgrid and flatten it to get a list of all coordinate pairs
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T

    # Convert degrees to radians for scikit-learn's haversine_distances function
    coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])

    # Earth radius in kilometers
    earth_radius_km = 6371.0

    logging.info(f"Calculating pairwise Haversine distances for {len(coords)} nodes...")
    distance_matrix = haversine_distances(coords_rad) * earth_radius_km
    logging.info(f"Calculated distance matrix with shape: {distance_matrix.shape}")

    return distance_matrix


def construct_binary_adjacency(distance_matrix: np.ndarray, distance_threshold_km: float = 150.0) -> np.ndarray:
    """
    Constructs a binary adjacency matrix based on a distance threshold.

    Args:
        distance_matrix (np.ndarray): The matrix of pairwise distances.
        distance_threshold_km (float): The distance threshold in kilometers.

    Returns:
        np.ndarray: The binary adjacency matrix.
    """
    logging.info(f"Constructing binary adjacency matrix with threshold {distance_threshold_km} km...")

    # Create a boolean matrix where True means the distance is within the threshold
    adj_matrix = (distance_matrix <= distance_threshold_km).astype(int)

    # Remove self-loops
    np.fill_diagonal(adj_matrix, 0)

    logging.info(f"Constructed binary adjacency matrix with {np.sum(adj_matrix)} edges.")
    return adj_matrix


def compute_degree_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the degree matrix from an adjacency matrix.

    Args:
        adj_matrix (np.ndarray): The binary adjacency matrix.

    Returns:
        np.ndarray: The degree matrix.
    """
    logging.info("Computing degree matrix...")
    degree_vector = np.sum(adj_matrix, axis=1)
    degree_matrix = np.diag(degree_vector)
    logging.info("Degree matrix computed.")
    return degree_matrix


def symmetrically_normalize_adjacency(adj_matrix: np.ndarray) -> coo_matrix:
    """
    Symmetrically normalizes the adjacency matrix.

    Args:
        adj_matrix (np.ndarray): The binary adjacency matrix.

    Returns:
        scipy.sparse.coo_matrix: The normalized sparse adjacency matrix.
    """
    logging.info("Symmetrically normalizing adjacency matrix...")

    # Using scipy.sparse for efficient matrix operations
    adj_sparse = coo_matrix(adj_matrix)

    # Compute D^(-1/2)
    degree_vector = np.array(adj_sparse.sum(axis=1)).flatten()
    # To avoid division by zero, we add a small epsilon where degree is 0,
    # then the inverse sqrt will be 0 anyway for those nodes.
    with np.errstate(divide="ignore"):
        inv_sqrt_degree_vector = 1.0 / np.sqrt(degree_vector)
    inv_sqrt_degree_vector[np.isinf(inv_sqrt_degree_vector)] = 0

    inv_sqrt_degree_matrix = diags(inv_sqrt_degree_vector)

    # D^(-1/2) * A * D^(-1/2)
    normalized_adj = inv_sqrt_degree_matrix.dot(adj_sparse).dot(inv_sqrt_degree_matrix)

    logging.info("Normalization complete.")
    return normalized_adj.tocoo()  # Return in COO format for easy edge extraction


def convert_to_pyg_and_save(normalized_adj: coo_matrix, output_path: str):
    """
    Converts a sparse adjacency matrix to PyTorch Geometric format and saves it.

    Args:
        normalized_adj (coo_matrix): The normalized sparse adjacency matrix.
        output_path (str): The path to save the output .pt file.
    """
    logging.info(f"Converting to PyG format and saving to {output_path}...")

    # Extract rows and columns for edge_index
    edge_index = torch.tensor(np.vstack((normalized_adj.row, normalized_adj.col)), dtype=torch.long)

    # The data attribute of the COO matrix contains the edge weights
    edge_weight = torch.tensor(normalized_adj.data, dtype=torch.float)

    # Save the tensors to a file
    torch.save({"edge_index": edge_index, "edge_weight": edge_weight}, output_path)

    logging.info(f"Graph data saved successfully. Edges: {edge_index.shape[1]}")


if __name__ == "__main__":
    logging.info("--- Running Test for Graph Constructor (Full Task 2) ---")

    # Define file paths
    files = ["data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5", "data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5"]

    # 1. Get Coordinates
    latitude, longitude = get_coordinates_from_data(files)

    if latitude is not None and longitude is not None:
        # 2. Calculate Distance Matrix
        dist_matrix = calculate_haversine_distance_matrix(latitude, longitude)

        # Verification
        num_nodes = 41 * 71
        assert dist_matrix.shape == (num_nodes, num_nodes), "Distance matrix shape is incorrect."
        logging.info("Test PASSED: Distance matrix shape is correct.")

        # Check for symmetry (with a small tolerance for floating point errors)
        assert np.allclose(dist_matrix, dist_matrix.T), "Distance matrix is not symmetric."
        logging.info("Test PASSED: Distance matrix is symmetric.")

        # Check for zero diagonal
        assert np.all(np.diag(dist_matrix) == 0), "Distance matrix diagonal is not all zeros."
        logging.info("Test PASSED: Distance matrix has a zero diagonal.")

        # 3. Construct Binary Adjacency Matrix
        adj_matrix = construct_binary_adjacency(dist_matrix)

        # Verification for 2.2
        assert adj_matrix.shape == dist_matrix.shape, "Adjacency matrix shape is incorrect."
        assert np.all((adj_matrix == 0) | (adj_matrix == 1)), "Adjacency matrix is not binary."
        assert np.all(np.diag(adj_matrix) == 0), "Adjacency matrix diagonal is not all zeros (self-loops exist)."
        logging.info("Test PASSED: Adjacency matrix is binary with no self-loops.")

        # 4. Compute Degree Matrix
        degree_matrix = compute_degree_matrix(adj_matrix)

        # Verification for 2.3
        assert degree_matrix.shape == adj_matrix.shape, "Degree matrix shape is incorrect."
        assert np.all(degree_matrix.diagonal() == np.sum(adj_matrix, axis=1)), "Degree matrix diagonals are not correct."
        # Check that off-diagonal elements are zero
        assert np.all(degree_matrix - np.diag(np.diag(degree_matrix)) == 0), "Degree matrix is not diagonal."
        logging.info("Test PASSED: Degree matrix is correct.")

        # 5. Symmetrically Normalize Adjacency Matrix
        normalized_adj_matrix = symmetrically_normalize_adjacency(adj_matrix)

        # Verification for 2.4
        assert normalized_adj_matrix.shape == adj_matrix.shape, "Normalized matrix shape is incorrect."
        # Check that the matrix is still symmetric
        assert np.allclose(normalized_adj_matrix.toarray(), normalized_adj_matrix.toarray().T), "Normalized matrix is not symmetric."
        # Check that values are between 0 and 1
        assert normalized_adj_matrix.min() >= 0 and normalized_adj_matrix.max() <= 1, "Normalized matrix values are out of [0, 1] range."
        logging.info("Test PASSED: Adjacency matrix normalized successfully.")

        # 6. Convert to PyG and Save
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "graph_A.pt")
        convert_to_pyg_and_save(normalized_adj_matrix, output_file_path)

        # Verification for 2.5
        assert os.path.exists(output_file_path), "Output file was not created."
        loaded_graph = torch.load(output_file_path, weights_only=False)
        assert "edge_index" in loaded_graph, "Loaded graph missing 'edge_index'."
        assert "edge_weight" in loaded_graph, "Loaded graph missing 'edge_weight'."
        assert loaded_graph["edge_index"].shape[0] == 2, "Loaded edge_index has incorrect shape."
        assert loaded_graph["edge_index"].shape[1] == normalized_adj_matrix.nnz, "Number of edges is incorrect."
        assert loaded_graph["edge_weight"].shape[0] == normalized_adj_matrix.nnz, "Number of edge weights is incorrect."
        logging.info("Test PASSED: Graph data saved and loaded correctly.")

        logging.info("--- Graph Constructor Test (Full Task 2) Finished ---")
    else:
        logging.error("Test FAILED: Could not retrieve coordinates.")
