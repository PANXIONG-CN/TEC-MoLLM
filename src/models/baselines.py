import numpy as np
import logging
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HistoricalAverage:
    def __init__(self):
        self.averages = None

    def fit(self, tec_data: np.ndarray, time_data: np.ndarray):
        """
        Calculates the historical average for each node and time-of-day slot.
        
        Args:
            tec_data (np.ndarray): Shape (N_times, N_nodes).
            time_data (np.ndarray): Corresponding time information with hour.
        """
        logging.info("Fitting Historical Average model...")
        num_nodes = tec_data.shape[1]
        # 2-hour resolution -> 12 slots per day
        self.averages = np.zeros((num_nodes, 12))
        
        hours = (time_data.astype('datetime64[h]').astype(int) % 24).astype(int)
        time_slots = hours // 2
        
        for node in range(num_nodes):
            for slot in range(12):
                mask = time_slots == slot
                self.averages[node, slot] = np.mean(tec_data[mask, node])
        logging.info("HA model fitted.")

    def predict(self, time_data: np.ndarray, num_nodes: int) -> np.ndarray:
        """
        Predicts using the pre-calculated averages.
        """
        hours = (time_data.astype('datetime64[h]').astype(int) % 24).astype(int)
        time_slots = hours // 2
        predictions = np.zeros((len(time_data), num_nodes))
        
        for i, slot in enumerate(time_slots):
            predictions[i, :] = self.averages[:, slot]
        return predictions

class SarimaBaseline:
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.models = {}
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, tec_data: np.ndarray, node_indices: list):
        """
        Fits a SARIMA model for each specified node.
        """
        for node_idx in node_indices:
            logging.info(f"Fitting SARIMA for node {node_idx}...")
            time_series = tec_data[:, node_idx]
            model = SARIMAX(time_series, order=self.order, seasonal_order=self.seasonal_order)
            self.models[node_idx] = model.fit(disp=False)
            logging.info(f"SARIMA fitted for node {node_idx}.")

    def predict(self, node_indices: list, steps: int) -> dict:
        """
        Makes predictions for specified nodes.
        """
        predictions = {}
        for node_idx in node_indices:
            if node_idx in self.models:
                predictions[node_idx] = self.models[node_idx].forecast(steps=steps)
        return predictions

def save_baseline(model, path):
    joblib.dump(model, path)
    logging.info(f"Baseline model saved to {path}")

def load_baseline(path):
    logging.info(f"Loading baseline model from {path}")
    return joblib.load(path) 