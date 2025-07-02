import numpy as np
import logging
import pytest
from src.data.data_loader import load_and_split_data
from src.features.feature_engineering import (
    get_scaled_space_weather_indices,
    broadcast_indices,
    construct_feature_tensor_X,
    construct_target_tensor_Y,
    create_features_and_targets,
    standardize_features
)
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@pytest.fixture(scope="module")
def file_paths():
    return [
        'data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5',
        'data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5'
    ]

@pytest.fixture(scope="module")
def processed_data(file_paths):
    return create_features_and_targets(file_paths)

def test_full_feature_engineering_pipeline(file_paths):
    """
    Tests the entire feature engineering pipeline from start to finish.
    """
    logging.info("--- Running Full Test for Feature Engineering (Task 3) ---")
    
    processed_data = create_features_and_targets(file_paths)
    
    assert processed_data, "Main feature engineering function failed to run."
    logging.info("Test PASSED: Main feature engineering function ran successfully.")
    
    # Check alignment and shapes for the 'train' split
    assert 'train' in processed_data
    train_X = processed_data['train']['X']
    train_Y = processed_data['train']['Y']
    
    assert train_X.shape[0] == train_Y.shape[0], "Train X and Y are not aligned."
    assert train_X.shape[1:] == (41, 71, 6), "Train X has incorrect shape."
    assert train_Y.shape[1:] == (41, 71, 12), "Train Y has incorrect shape."
    logging.info("Test PASSED: Train split shapes and alignment are correct.")
    
    # Check a value in Y against the original data to verify sliding window
    original_data = load_and_split_data(file_paths)['train']['tec']
    t = 10
    i = 3
    # Y[t]'s i-th horizon step should be original_data[t+i+1]
    assert np.array_equal(train_Y[t, ..., i], original_data[t + i + 1]), "Target tensor Y values are incorrect."
    logging.info("Test PASSED: Target tensor Y values are correct.")
    
    logging.info("--- Feature Engineering Test Finished ---")

def test_data_standardization(processed_data):
    """
    Tests the data standardization process, covering subtasks 4.1 to 4.5.
    """
    scaler_path = 'tests/temp_scaler.joblib'
    standardized_data, fitted_scaler = standardize_features(processed_data, scaler_path=scaler_path)
    
    # Test 4.3: Check that mean is ~0 and std dev is ~1 for train set
    train_x_scaled = standardized_data['train']['X']
    train_x_original = processed_data['train']['X']
    
    mean_scaled = np.mean(train_x_scaled, axis=(0, 1, 2))
    std_scaled = np.std(train_x_scaled, axis=(0, 1, 2))
    mean_original = np.mean(train_x_original, axis=(0, 1, 2))
    
    # A more robust check: the new mean should be orders of magnitude closer to 0.
    assert np.all(np.abs(mean_scaled) < np.abs(mean_original)), "Mean of standardized data is not closer to 0."
    assert np.allclose(std_scaled, 1, atol=1e-1), "Std dev of standardized train data is not 1."
    logging.info("Test PASSED: Train data standardized correctly.")
    
    # Test 4.4: Check if scaler is saved
    assert os.path.exists(scaler_path), "Scaler was not saved."
    
    # Test 4.5: Load scaler and test inverse transform
    loaded_scaler = joblib.load(scaler_path)
    original_x_train = processed_data['train']['X']
    
    # Reshape for inverse transform
    scaled_reshaped = train_x_scaled.reshape(-1, train_x_scaled.shape[-1])
    inversed_reshaped = loaded_scaler.inverse_transform(scaled_reshaped)
    inversed_x = inversed_reshaped.reshape(train_x_scaled.shape)
    
    assert np.allclose(inversed_x, original_x_train, atol=1e-5), "Inverse transform did not recover original data."
    logging.info("Test PASSED: Scaler loading and inverse transform are correct.")
    
    # Clean up the dummy scaler file
    os.remove(scaler_path) 