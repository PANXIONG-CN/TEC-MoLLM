# TEC-MoLLM: Total Electron Content Prediction using Multi-modal Large Language Model

A hybrid deep learning model for predicting Total Electron Content (TEC) in the ionosphere, combining Graph Neural Networks, Temporal Convolutional Networks, and Large Language Models with LoRA fine-tuning.

## 🌟 Features

- **Multi-modal Architecture**: Integrates spatial, temporal, and contextual features
- **Graph Neural Networks**: Captures geographical relationships using GATv2
- **Temporal Modeling**: Multi-scale convolutional embedding inspired by SeisMoLLM
- **LLM Integration**: GPT-2 backbone with LoRA for parameter-efficient fine-tuning
- **Comprehensive Evaluation**: Multiple metrics including MAE, RMSE, R², and Pearson correlation

## 🏗️ Architecture

```
Input Features (TEC + Space Weather) 
    ↓
Spatio-Temporal Embedding
    ↓
Spatial Encoder (GATv2)
    ↓
Temporal Encoder (Multi-scale Conv + Patching)
    ↓
LLM Backbone (GPT-2 + LoRA)
    ↓
Prediction Head
    ↓
TEC Predictions (12 horizons)
```

## 📊 Dataset

- **Data Source**: CRIM_SW2hr_AI ionosphere data (2014-2015)
- **Features**: 
  - TEC values (41×71 spatial grid)
  - Space weather indices (5 parameters)
  - Temporal coordinates
- **Splits**: 
  - Training: 2014 (4,368 samples)
  - Validation: Jan-Jun 2015 (2,160 samples) 
  - Test: Jul-Dec 2015 (2,196 samples)

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torch-geometric transformers peft einops
pip install scikit-learn scipy h5py joblib tqdm
```

### Training

```bash
python train.py
```

### Project Structure

```
TEC-MoLLM/
├── src/
│   ├── data/
│   │   ├── data_loader.py      # HDF5 data loading
│   │   └── dataset.py          # PyTorch sliding window dataset
│   ├── features/
│   │   └── feature_engineering.py  # Feature preprocessing
│   ├── graph/
│   │   └── graph_constructor.py     # Spatial graph construction
│   ├── model/
│   │   ├── modules.py               # Core model components
│   │   └── tec_mollm.py            # Main model assembly
│   ├── evaluation/
│   │   └── metrics.py              # Evaluation metrics
│   └── models/
│       └── baselines.py            # Baseline models (HA, SARIMA)
├── tests/                          # Unit tests
├── train.py                       # Training script
├── test.py                        # Testing script
└── data/
    ├── raw/                       # Original HDF5 files
    └── processed/                 # Preprocessed data
```

## 📈 Results

### Current Performance (3 epochs, subset data)

| Metric | Value |
|--------|-------|
| MAE | 197.03 TECU |
| RMSE | 285.82 TECU |
| R² | 0.668 |
| Pearson R | 0.820 |

### Training Progress

- **Epoch 1**: Train Loss: 468.45 → Val Loss: 198.98
- **Epoch 2**: Train Loss: 221.73 → Val Loss: 89.52
- **Epoch 3**: Continued improvement

## 🔧 Model Components

### 1. Spatio-Temporal Embedding
- Node embeddings for 2,911 grid points
- Time-of-day and day-of-year embeddings
- Learnable 16-dimensional representations

### 2. Spatial Encoder (GATv2)
- Graph attention mechanism for spatial dependencies
- Edge connectivity based on geographical distance
- Multi-head attention (2 heads, 32 output channels)

### 3. Temporal Encoder
- Multi-scale convolutional blocks (kernel sizes: 3, 5, 7)
- Downsampling with strides [2, 2]: 24 → 12 → 6 timesteps
- Latent patching for LLM compatibility

### 4. LLM Backbone
- GPT-2 model truncated to 3 layers
- LoRA adaptation (r=16, α=32)
- Parameter-efficient fine-tuning (1.55% trainable parameters)

### 5. Prediction Head
- Linear projection to 12 prediction horizons
- Maps from LLM hidden dimension to TEC forecasts

## 🎯 Future Improvements

### High Priority
- [ ] Use full dataset (4,368 training samples)
- [ ] Extend training to 50-100 epochs
- [ ] Implement real time features (hour/day_of_year)
- [ ] Increase input window size to 72-168 timesteps

### Medium Priority
- [ ] Advanced loss functions (Huber, MAE)
- [ ] Learning rate scheduling
- [ ] Data augmentation strategies
- [ ] Deeper LLM (6-12 layers)

### Low Priority
- [ ] Attention mechanisms for spatio-temporal fusion
- [ ] Physics-informed constraints
- [ ] Model ensemble methods
- [ ] Hyperparameter optimization

## 📚 References

- **Graph Attention Networks**: Veličković et al. (2018)
- **LoRA**: Hu et al. (2021) - Low-Rank Adaptation of Large Language Models
- **SeisMoLLM**: Inspired by seismic modeling approaches
- **TEC Prediction**: Ionosphere research and space weather modeling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

For questions and collaborations, please open an issue in this repository.

---

⭐ **Star this repository if you find it helpful!**