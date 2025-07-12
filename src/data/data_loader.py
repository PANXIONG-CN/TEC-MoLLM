import h5py
import numpy as np
import logging
import pandas as pd
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_data_from_hdf5(file_path: str) -> dict:
    """
    (Internal) Reads specified datasets from a single HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        dict: A dictionary containing the datasets. Returns an empty dictionary if an error occurs.
    """
    logging.info(f"Attempting to load data from {file_path}")
    data = {}
    try:
        with h5py.File(file_path, 'r') as f:
            if 'ionosphere/TEC' in f:
                data['tec'] = f['ionosphere/TEC'][:]
                logging.info(f"Loaded 'tec' with shape: {data['tec'].shape}")
            else:
                logging.error("'ionosphere/TEC' not found in the file.")
                return {}

            # Correct path for time information based on h5ls inspection
            if 'coordinates/datetime_utc' in f:
                data['time'] = f['coordinates/datetime_utc'][:]
                logging.info(f"Loaded 'time' with shape: {data['time'].shape}")
            else:
                logging.error("'coordinates/datetime_utc' not found. Time-based split will fail.")
                return {}

            if 'space_weather_indices' in f:
                # This is a group, not a dataset. We need to stack the individual indices.
                try:
                    ae = f['space_weather_indices/AE_Index'][:]
                    dst = f['space_weather_indices/Dst_Index'][:]
                    f107 = f['space_weather_indices/F107_Index'][:]
                    
                    # Handle Kp Index with proper scale_factor
                    kp_raw = f['space_weather_indices/Kp_Index'][:]
                    kp_scale_factor = f['space_weather_indices/Kp_Index'].attrs.get('scale_factor', 1.0)
                    kp = kp_raw * kp_scale_factor
                    logging.info(f"Applied scale_factor {kp_scale_factor} to Kp_Index")
                    
                    ap = f['space_weather_indices/ap_Index'][:]
                    
                    # Stack them into a single array (T, num_indices)
                    data['space_weather_indices'] = np.stack([ae, dst, f107, kp, ap], axis=-1)
                    logging.info(f"Loaded and stacked 'space_weather_indices' with shape: {data['space_weather_indices'].shape}")
                except KeyError as e:
                    logging.error(f"Could not find an index within space_weather_indices group: {e}")
            else:
                logging.warning("'space_weather_indices' group not found.")

            if 'coordinates' in f:
                # We need specific coordinate arrays, not the whole group
                if 'coordinates/latitude' in f and 'coordinates/longitude' in f:
                    data['latitude'] = f['coordinates/latitude'][:]
                    data['longitude'] = f['coordinates/longitude'][:]
                    logging.info(f"Loaded 'latitude' with shape: {data['latitude'].shape}")
                    logging.info(f"Loaded 'longitude' with shape: {data['longitude'].shape}")
                else:
                    logging.warning("Latitude/Longitude not found under /coordinates.")
            else:
                logging.warning("'coordinates' group not found.")

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}
    except Exception as e:
        logging.error(f"An error occurred while reading {file_path}: {e}")
        return {}
    
    logging.info(f"Successfully loaded data from {file_path}")
    return data

def _aggregate_data(file_paths: list) -> dict:
    """
    (Internal) Loads data from a list of HDF5 files and aggregates them.

    Args:
        file_paths (list): A list of paths to the HDF5 files.

    Returns:
        dict: A dictionary containing the aggregated datasets.
    """
    all_data = []
    for path in file_paths:
        data = _load_data_from_hdf5(path)
        if data:
            all_data.append(data)
    
    if not all_data:
        logging.error("No data could be loaded from any file paths.")
        return {}

    # Assuming all files have the same keys and structure
    aggregated = {}
    keys_to_aggregate = ['tec', 'time', 'space_weather_indices']
    
    for key in keys_to_aggregate:
        # Concatenate along the time axis (axis=0)
        aggregated[key] = np.concatenate([d[key] for d in all_data if key in d], axis=0)
        logging.info(f"Aggregated '{key}' with final shape: {aggregated[key].shape}")

    # Convert byte strings to datetime objects for proper comparison and manipulation
    try:
        # The time data is in byte format (e.g., b'2014-01-01T00:00:00'), decode it first
        decoded_time = np.char.decode(aggregated['time'])
        aggregated['time'] = pd.to_datetime(decoded_time)
        logging.info("Successfully converted 'time' array to datetime objects.")
    except Exception as e:
        logging.error(f"Failed to convert time strings to datetime objects: {e}")
        return {}

    # Coordinates are static and should be the same across files, so just take from the first file.
    static_keys = ['latitude', 'longitude']
    for key in static_keys:
        if key in all_data[0]:
            aggregated[key] = all_data[0][key]
            logging.info(f"Took static key '{key}' from first file with shape: {aggregated[key].shape}")

    return aggregated

def _split_data(aggregated_data: dict) -> dict:
    """
    (Internal) Splits the aggregated data into training, validation, and test sets based on date ranges.

    Args:
        aggregated_data (dict): A dictionary containing the aggregated datasets, including a 'time'
                                  key with datetime objects.

    Returns:
        dict: A dictionary containing the data splits, e.g., {'train': {...}, 'val': {...}, 'test': {...}}.
    """
    if 'time' not in aggregated_data:
        logging.error("Cannot split data without 'time' key.")
        return {}
    
    # Create a pandas Series for easy boolean indexing with dates
    timestamps = pd.Series(aggregated_data['time'])

    # --- START MODIFICATION ---
    # 假设您的13年数据是从 2013年初 到 2025年底
    # 9年训练: 2013-2021
    # 2年验证: 2022-2023
    # ~1.4年测试: 2024.01-2025.04
    # 请根据您的实际数据起止年份进行调整
    train_end = '2021-12-31 23:59:59'
    val_start = '2022-01-01 00:00:00'
    val_end = '2023-12-31 23:59:59'
    test_start = '2024-01-01 00:00:00'
    # --- END MODIFICATION ---

    # Create boolean masks
    train_mask = timestamps <= train_end
    val_mask = (timestamps >= val_start) & (timestamps <= val_end)
    test_mask = timestamps >= test_start

    data_splits = {}
    for split_name, mask in zip(['train', 'val', 'test'], [train_mask, val_mask, test_mask]):
        split = {}
        for key, value in aggregated_data.items():
            if hasattr(value, 'ndim') and value.ndim > 1 or key == 'time': # Split time-series data
                split[key] = value[mask]
            else: # Keep static data like coordinates
                split[key] = value
        data_splits[split_name] = split
        logging.info(f"Created '{split_name}' split with {len(split['time'])} samples.")
        
    return data_splits

def load_and_split_data(file_paths: List[str]) -> Dict[str, Any]:
    """
    Main function to load, aggregate, and split data from HDF5 files.

    This is the primary public function of this module.

    Args:
        file_paths (List[str]): List of paths to the HDF5 files.

    Returns:
        Dict[str, Any]: A dictionary containing the 'train', 'val', and 'test' data splits.
    """
    logging.info("Starting data loading and splitting process...")
    
    # Step 1: Aggregate data from all files
    aggregated_data = _aggregate_data(file_paths)
    if not aggregated_data:
        logging.error("Aggregation step failed. Aborting.")
        return {}
    
    # Step 2: Split the aggregated data
    data_splits = _split_data(aggregated_data)
    if not data_splits:
        logging.error("Splitting step failed. Aborting.")
        return {}
        
    logging.info("Data loading and splitting process completed successfully.")
    return data_splits

if __name__ == '__main__':
    # This block now serves as a final integration test and usage example.
    logging.info("--- Running Master Data Loading Script Test ---")
    
    # Define file paths
    files = [
        'data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5',
        'data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5'
    ]
    
    # Run the main function
    final_data = load_and_split_data(files)
    
    # Verification
    if final_data:
        logging.info("Test PASSED: Master script ran successfully.")
        assert 'train' in final_data and 'val' in final_data and 'test' in final_data
        
        # Check sample counts
        assert len(final_data['train']['time']) == 4380
        assert len(final_data['val']['time']) == 2172
        assert len(final_data['test']['time']) == 2208
        logging.info("Test PASSED: Final splits have the correct number of samples.")
        
        # Check shapes
        logging.info(f"Train TEC shape: {final_data['train']['tec'].shape}")
        logging.info(f"Val TEC shape: {final_data['val']['tec'].shape}")
        logging.info(f"Test TEC shape: {final_data['test']['tec'].shape}")
    else:
        logging.error("Test FAILED: Master script failed to produce data.")

    logging.info("--- Master Test Finished ---") 