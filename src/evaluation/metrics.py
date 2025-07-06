import numpy as np
import logging
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler) -> dict:
    """
    Calculates multiple evaluation metrics by first applying inverse transform to scaled data.
    
    Args:
        y_true_scaled (np.ndarray): The ground truth values, scaled.
        y_pred_scaled (np.ndarray): The predicted values, scaled.
        scaler: The fitted scaler for inverse transformation.
        
    Returns:
        dict: A dictionary of calculated metrics.
    """
    # 在计算指标前，先进行逆变换
    original_true_shape = y_true_scaled.shape
    original_pred_shape = y_pred_scaled.shape
    
    # Reshape for scaler (expects 2D input)
    y_true_reshaped = y_true_scaled.reshape(-1, 1)
    y_pred_reshaped = y_pred_scaled.reshape(-1, 1)
    
    # Apply inverse transform
    y_true_unscaled_reshaped = scaler.inverse_transform(y_true_reshaped)
    y_pred_unscaled_reshaped = scaler.inverse_transform(y_pred_reshaped)
    
    # Reshape back to original dimensions
    y_true_unscaled = y_true_unscaled_reshaped.reshape(original_true_shape)
    y_pred_unscaled = y_pred_unscaled_reshaped.reshape(original_pred_shape)

    if y_true_unscaled.ndim > 2: # Reshape if data is in (B, L, N, C) format
        y_true_unscaled = y_true_unscaled.reshape(-1, y_true_unscaled.shape[-1])
        y_pred_unscaled = y_pred_unscaled.reshape(-1, y_pred_unscaled.shape[-1])

    # 后续所有计算都在unscaled值上进行
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
    r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    
    # --- START MODIFICATION 1.4.1 ---
    # REASON: The original Pearson R calculation was flawed for single-feature predictions.
    #         This version correctly calculates it on the flattened data arrays.
    # OLD Pearson R block:
    # pearson_coeffs = []
    # for i in range(y_true_unscaled.shape[1]): ...
    # avg_pearson_r = np.mean(pearson_coeffs)

    # NEW Pearson R calculation:
    # Ensure arrays are 1D for pearsonr
    y_true_flat = y_true_unscaled.ravel()
    y_pred_flat = y_pred_unscaled.ravel()
    if np.std(y_true_flat) > 0 and np.std(y_pred_flat) > 0:
        avg_pearson_r = pearsonr(y_true_flat, y_pred_flat)[0]
    else:
        avg_pearson_r = 0.0
    # --- END MODIFICATION 1.4.1 ---


    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'pearson_r': avg_pearson_r
    }
    
    return metrics

def evaluate_metrics_unscaled_fallback(y_true_unscaled: np.ndarray, y_pred_unscaled: np.ndarray) -> dict:
    """
    Fallback function for calculating metrics on already unscaled data (backward compatibility).
    """
    if y_true_unscaled.ndim > 2:
        y_true_unscaled = y_true_unscaled.reshape(-1, y_true_unscaled.shape[-1])
        y_pred_unscaled = y_pred_unscaled.reshape(-1, y_pred_unscaled.shape[-1])

    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
    r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    
    pearson_coeffs = []
    for i in range(y_true_unscaled.shape[1]):
        if np.std(y_true_unscaled[:, i]) > 0 and np.std(y_pred_unscaled[:, i]) > 0:
            pearson_coeffs.append(pearsonr(y_true_unscaled[:, i], y_pred_unscaled[:, i])[0])
        else:
            pearson_coeffs.append(0.0)
    
    avg_pearson_r = np.mean(pearson_coeffs)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'pearson_r': avg_pearson_r
    }

def evaluate_horizons(y_true_horizons_scaled: np.ndarray, y_pred_horizons_scaled: np.ndarray, target_scaler_path: str = None) -> dict:
    """
    Evaluates metrics across multiple prediction horizons by first applying inverse transform.
    
    Args:
        y_true_horizons_scaled (np.ndarray): Ground truth of shape (N, L_out, ...), scaled.
        y_pred_horizons_scaled (np.ndarray): Predictions of shape (N, L_out, ...), scaled.
        target_scaler_path (str): Path to the target scaler for inverse transformation.
        
    Returns:
        dict: A dictionary of average metrics across all horizons.
    """
    logging.info("Evaluating metrics across all horizons...")

    # --- START MODIFICATION 1.4.2 ---
    # REASON: Adds a defensive check to handle non-finite values (inf/nan) that
    #         were causing the 'overflow' warning. This prevents crashes and
    #         provides clear logging when numerical instability occurs.
    if not np.all(np.isfinite(y_pred_horizons_scaled)):
        num_non_finite = np.sum(~np.isfinite(y_pred_horizons_scaled))
        logging.warning(
            f"Overflow Guard: Found {num_non_finite} non-finite values in predictions. "
            "Clamping them to 0 for metric calculation."
        )
        y_pred_horizons_scaled = np.nan_to_num(y_pred_horizons_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    # --- END MODIFICATION 1.4.2 ---

    # Load scaler if provided
    if target_scaler_path:
        scaler = joblib.load(target_scaler_path)
        logging.info(f"Loaded target scaler from {target_scaler_path}")
    else:
        logging.warning("No target_scaler_path provided, assuming data is already unscaled")
        scaler = None

    all_horizon_metrics = []
    num_horizons = y_true_horizons_scaled.shape[1]

    for i in range(num_horizons):
        y_true_h = y_true_horizons_scaled[:, i]
        y_pred_h = y_pred_horizons_scaled[:, i]
        
        # Use the wrapper for a single horizon, passing scaler
        if scaler is not None:
            horizon_metrics = evaluate_metrics(y_true_h, y_pred_h, scaler)
        else:
            # Fallback to old behavior if no scaler provided (for backward compatibility)
            horizon_metrics = evaluate_metrics_unscaled_fallback(y_true_h, y_pred_h)
        all_horizon_metrics.append(horizon_metrics)
        
    # Calculate and report average metrics
    avg_metrics = {
        'mae_avg': np.mean([m['mae'] for m in all_horizon_metrics]),
        'rmse_avg': np.mean([m['rmse'] for m in all_horizon_metrics]),
        'r2_score_avg': np.mean([m['r2_score'] for m in all_horizon_metrics]),
        'pearson_r_avg': np.mean([m['pearson_r'] for m in all_horizon_metrics]),
        # Add detailed metrics by horizon
        'mae_by_horizon': [m['mae'] for m in all_horizon_metrics],
        'rmse_by_horizon': [m['rmse'] for m in all_horizon_metrics],
        'r2_by_horizon': [m['r2_score'] for m in all_horizon_metrics],
        'pearson_by_horizon': [m['pearson_r'] for m in all_horizon_metrics]
    }
    
    logging.info(f"Average metrics: {avg_metrics}")
    return avg_metrics 