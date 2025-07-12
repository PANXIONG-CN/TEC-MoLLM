import numpy as np
import pandas as pd
import logging
from src.data.data_loader import load_and_split_data
import joblib
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_scaled_space_weather_indices(data: dict) -> dict:
    """
    Extracts and scales space weather indices from the loaded data dictionary.
    """
    logging.info("Extracting and scaling space weather indices...")
    if 'space_weather_indices' not in data:
        logging.error("'space_weather_indices' not found in data.")
        return {}
    indices_data = data['space_weather_indices']
    index_names = ['AE_Index', 'Dst_Index', 'F107_Index', 'Kp_Index', 'ap_Index']
    scaled_indices = {}
    for i, name in enumerate(index_names):
        scale_factor = 1.0
        scaled_indices[name] = indices_data[:, i] * scale_factor
    return scaled_indices

def broadcast_indices(scaled_indices: dict, target_shape: tuple) -> dict:
    """
    Broadcasts the 1D space weather indices to match the spatial dimensions of the TEC data.
    """
    logging.info(f"Broadcasting indices to target shape {target_shape}...")
    broadcasted_indices = {}
    for name, index_array in scaled_indices.items():
        reshaped_array = index_array[:, np.newaxis, np.newaxis]
        broadcasted_indices[name] = np.broadcast_to(reshaped_array, (index_array.shape[0],) + target_shape)
    return broadcasted_indices

def construct_feature_tensor_X(tec_data: np.ndarray, broadcasted_indices: dict) -> np.ndarray:
    """
    Constructs the feature tensor X by stacking TEC data and broadcasted indices.
    """
    logging.info("Constructing feature tensor X...")
    index_names = ['AE_Index', 'Dst_Index', 'F107_Index', 'Kp_Index', 'ap_Index']
    arrays_to_stack = [tec_data[..., np.newaxis]]
    for name in index_names:
        if name in broadcasted_indices:
            arrays_to_stack.append(broadcasted_indices[name][..., np.newaxis])
        else:
            logging.error(f"Index '{name}' not found in broadcasted indices. Aborting.")
            return np.array([])
    feature_tensor_X = np.concatenate(arrays_to_stack, axis=-1)
    logging.info(f"Constructed feature tensor X with shape: {feature_tensor_X.shape}")
    return feature_tensor_X

def construct_target_tensor_Y(tec_data: np.ndarray, horizon: int = 12) -> np.ndarray:
    """
    Constructs the multi-step target tensor Y using a sliding window.
    """
    logging.info(f"Constructing target tensor Y with horizon {horizon}...")
    num_samples = tec_data.shape[0]
    num_targets = num_samples - horizon
    target_tensor_Y = np.zeros((num_targets, tec_data.shape[1], tec_data.shape[2], horizon))
    for i in range(num_targets):
        target_slice = tec_data[i+1 : i+1+horizon]
        target_tensor_Y[i] = target_slice.transpose(1, 2, 0)
    logging.info(f"Constructed target tensor Y with shape: {target_tensor_Y.shape}")
    return target_tensor_Y

def extract_time_features(time_data: pd.DatetimeIndex) -> np.ndarray:
    """
    Extract time features from datetime data.
    
    Args:
        time_data (pd.DatetimeIndex): Datetime index array
        
    Returns:
        np.ndarray: Array of shape (N, 4) containing 
                   [time_of_day_slot, day_of_year, year_index, season_index]
    """
    logging.info("Extracting extended time features (TOD, DOY, Year, Season)...")
    # 1. Time of day slot (0-11)
    time_of_day_slot = time_data.hour // 2
    
    # 2. Day of year (0-365)
    day_of_year = time_data.dayofyear - 1
    
    # --- START MODIFICATION ---
    # 3. Year index (e.g., 2014 -> 0, 2015 -> 1, ...)
    #    This makes the embedding independent of the absolute year value.
    start_year = time_data.year.min()
    year_index = time_data.year - start_year

    # 4. Season index (0:Winter, 1:Spring, 2:Summer, 3:Autumn)
    #    Maps month to season index: (12,1,2)->0, (3,4,5)->1, (6,7,8)->2, (9,10,11)->3
    season_index = (time_data.month % 12 + 3) // 3 - 1
    # --- END MODIFICATION ---
    
    # Stack into (N, 4) array
    time_features = np.stack([time_of_day_slot, day_of_year, year_index, season_index], axis=-1)
    logging.info(f"Extracted time features with shape: {time_features.shape}")
    
    return time_features

def create_features_and_targets(file_paths: list, horizon: int = 12) -> dict:
    """
    Main function to perform feature and target engineering for a given data split.
    """
    logging.info("Starting feature and target engineering process...")
    data_splits = load_and_split_data(file_paths)
    if not data_splits:
        logging.error("Could not load data. Aborting feature engineering.")
        return {}
    processed_splits = {}
    for split_name, data in data_splits.items():
        logging.info(f"--- Processing split: {split_name} ---")
        scaled_indices = get_scaled_space_weather_indices(data)
        broadcasted = broadcast_indices(scaled_indices, data['tec'].shape[1:])
        
        # --- START MODIFICATION 1.4.1 (Data Transformation) ---
        # REASON: Using direct physical space standardization instead of log transform
        #         for simpler and more interpretable processing pipeline.
        tec_data = data['tec']
        # No transformation applied - work directly with physical TEC values
        feature_tensor_X = construct_feature_tensor_X(tec_data, broadcasted)
        target_tensor_Y = construct_target_tensor_Y(tec_data, horizon)
        # --- END MODIFICATION 1.4.1 ---
        
        # Extract real time features from datetime data
        time_features = extract_time_features(data['time'])
        
        num_targets = target_tensor_Y.shape[0]
        aligned_X = feature_tensor_X[:num_targets]
        aligned_time_features = time_features[:num_targets]
        
        logging.info(f"Aligned X shape: {aligned_X.shape}, Aligned Y shape: {target_tensor_Y.shape}")
        logging.info(f"Aligned time features shape: {aligned_time_features.shape}")
        
        processed_splits[split_name] = {
            'X': aligned_X, 
            'Y': target_tensor_Y,
            'time_features': aligned_time_features
        }
    logging.info("Feature and target engineering completed.")
    return processed_splits

def standardize_features(processed_splits: dict, scaler_path: str = 'data/processed/scaler.joblib') -> (dict, StandardScaler):
    """
    Standardizes the feature tensor X for all data splits.
    
    Args:
        processed_splits (dict): The dictionary containing 'X' and 'Y' for each split.
        scaler_path (str): Path to save the fitted scaler.
        
    Returns:
        tuple: A tuple containing:
            - dict: The dictionary with standardized 'X' tensors.
            - StandardScaler: The fitted scaler object.
    """
    logging.info("Standardizing features...")
    
    X_train = processed_splits['train']['X']
    original_shape_train = X_train.shape
    
    # Subtask 4.1: Reshape for scaler
    X_train_reshaped = X_train.reshape(-1, original_shape_train[-1])
    logging.info(f"Reshaped X_train for scaler: {X_train_reshaped.shape}")
    
    # Subtask 4.2: Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train_reshaped)
    logging.info("StandardScaler fitted on training data.")
    
    # Subtask 4.4: Save the scaler
    import os
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")
    
    # Subtask 4.3: Apply transformation
    standardized_splits = {}
    for split_name, data in processed_splits.items():
        X = data['X']
        original_shape = X.shape
        X_reshaped = X.reshape(-1, original_shape[-1])
        
        X_scaled_reshaped = scaler.transform(X_reshaped)
        
        # Reshape back to original dimensions
        X_scaled = X_scaled_reshaped.reshape(original_shape)
        
        standardized_splits[split_name] = {'X': X_scaled, 'Y': data['Y']}
        logging.info(f"Standardized '{split_name}' X with shape {X_scaled.shape}")
        
    return standardized_splits, scaler

if __name__ == '__main__':
    logging.info("--- Running Test for Feature Engineering (Task 3) ---")
    files = [
        'data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5',
        'data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5'
    ]
    processed_data = create_features_and_targets(files)
    if processed_data:
        logging.info("Test PASSED: Main feature engineering function ran successfully.")
        train_X = processed_data['train']['X']
        train_Y = processed_data['train']['Y']
        assert train_X.shape[0] == train_Y.shape[0], "Train X and Y are not aligned."
        assert train_X.shape[3] == 6, "Train X has incorrect number of features."
        assert train_Y.shape[3] == 12, "Train Y has incorrect horizon."
        logging.info("Test PASSED: Train split shapes and alignment are correct.")
        original_data = load_and_split_data(files)['train']['tec']
        t = 0
        i = 5
        assert np.array_equal(train_Y[t, ..., i], original_data[t+i+1]), "Target tensor Y values are incorrect."
        logging.info("Test PASSED: Target tensor Y values are correct.")
    else:
        logging.error("Test FAILED: Feature engineering process failed.")
    logging.info("--- Feature Engineering Test Finished ---")
