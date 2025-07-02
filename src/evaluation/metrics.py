import numpy as np
import logging
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler: joblib.memory.MemorizedResult) -> dict:
    """
    Calculates multiple evaluation metrics after inverse transforming the data.
    
    Args:
        y_true (np.ndarray): The ground truth values, scaled.
        y_pred (np.ndarray): The predicted values, scaled.
        scaler (joblib.memory.MemorizedResult): The fitted scaler object.
        
    Returns:
        dict: A dictionary of calculated metrics.
    """
    if y_true.ndim > 2: # Reshape if data is in (B, L, N, C) format
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    # Inverse transform to get original scale
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)

    # Subtask 12.1: Core metric functions
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
    r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    
    # Pearson R needs to be calculated for each feature/horizon and then averaged
    pearson_coeffs = [pearsonr(y_true_unscaled[:, i], y_pred_unscaled[:, i])[0] for i in range(y_true_unscaled.shape[1])]
    avg_pearson_r = np.mean(pearson_coeffs)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'pearson_r': avg_pearson_r
    }
    
    return metrics

def evaluate_horizons(y_true_horizons: np.ndarray, y_pred_horizons: np.ndarray, scaler_path: str) -> dict:
    """
    Evaluates metrics across multiple prediction horizons.
    
    Args:
        y_true_horizons (np.ndarray): Ground truth of shape (N, L_out, ...).
        y_pred_horizons (np.ndarray): Predictions of shape (N, L_out, ...).
        scaler_path (str): Path to the saved scaler object.
        
    Returns:
        dict: A dictionary of average metrics across all horizons.
    """
    logging.info("Evaluating metrics across all horizons...")
    # Subtask 12.2: Load scaler
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        logging.error(f"Scaler file not found at {scaler_path}")
        return {}

    all_horizon_metrics = []
    num_horizons = y_true_horizons.shape[1]

    # Subtask 12.4: Adapt for multiple horizons
    for i in range(num_horizons):
        y_true_h = y_true_horizons[:, i]
        y_pred_h = y_pred_horizons[:, i]
        
        # Subtask 12.3: Use the wrapper for a single horizon
        horizon_metrics = evaluate_metrics(y_true_h, y_pred_h, scaler)
        all_horizon_metrics.append(horizon_metrics)
        
    # Subtask 12.5: Calculate and report average metrics
    avg_metrics = {
        'mae_avg': np.mean([m['mae'] for m in all_horizon_metrics]),
        'rmse_avg': np.mean([m['rmse'] for m in all_horizon_metrics]),
        'r2_score_avg': np.mean([m['r2_score'] for m in all_horizon_metrics]),
        'pearson_r_avg': np.mean([m['pearson_r'] for m in all_horizon_metrics])
    }
    
    logging.info(f"Average metrics: {avg_metrics}")
    return avg_metrics 