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
# Core dependencies
pip install torch torch-geometric transformers peft einops
pip install scikit-learn scipy h5py joblib tqdm pandas

# For distributed training
pip install torch torchrun
```

### 1. Data Preparation

First, build the spatial graph:

```bash
python src/graph/graph_constructor.py
```

This creates a spatial adjacency matrix with 2,911 nodes (41×71 grid) and 20,924 edges using a 150km distance threshold.

### 2. Training

#### Single GPU Training

```bash
python train.py --epochs 50 --batch_size 4 --L_in 48
```

#### Distributed Training (Recommended for 2x RTX 3090)

```bash
# Launch distributed training on 2 GPUs
torchrun --nproc_per_node 2 train.py --epochs 50 --batch_size 2 --L_in 48

# Monitor training progress - detailed metrics every 10 epochs
# Epoch 10/50: Detailed metrics with per-horizon breakdown
# Epoch 20/50: Comprehensive performance analysis
# ...
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size per GPU (default: 4, adjust based on GPU memory)
- `--L_in`: Input sequence length (default: 48 for memory efficiency)
- `--L_out`: Prediction horizon (default: 12)
- `--lr`: Learning rate (default: 1e-4)

### 3. Model Evaluation

```bash
# Evaluate trained model against baselines
python test.py --model_checkpoint checkpoints/best_model.pth --L_in 48

# Results will be saved to:
# - results/evaluation_results.csv (detailed metrics)
# - results/evaluation_summary.txt (summary report)
```

### 4. Training Monitoring

The training script provides comprehensive monitoring:

- **Every Epoch**: Basic train/validation loss and key metrics
- **Every 10 Epochs**: Detailed breakdown with:
  - Per-horizon MAE, RMSE, R², Pearson R values
  - Visual formatting with 80-character dividers
  - Best model checkpointing indicators

**Sample Output:**
```
================================================================================
DETAILED METRICS - Epoch 20/50
================================================================================
Training Loss: 45.234567
Validation Loss: 52.876543
Validation Metrics by Horizon:
  - MAE Average: 156.789012
  - RMSE Average: 223.456789
  - R² Score Average: 0.678901
  - Pearson R Average: 0.834567

MAE by Horizon:
  Horizon  1: 145.123456
  Horizon  2: 152.789012
  ...
  Horizon 12: 167.890123
================================================================================
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

### Current Performance (20 epochs, distributed training on 2x RTX 3090)

| Metric | Value |
|--------|-------|
| MAE | ~150-170 TECU |
| RMSE | ~220-250 TECU |
| R² | 0.67+ |
| Pearson R | 0.84+ |

### Training Progress

The distributed training framework shows excellent scalability and performance:

- **Distributed Setup**: Successfully implemented PyTorch DDP for 2x RTX 3090 GPUs
- **Memory Optimization**: L_in=48, batch_size=2 per GPU for stable training
- **Monitoring Enhancement**: Detailed metrics every 10 epochs with per-horizon breakdown
- **Performance Trajectory**: Consistent improvement from R² ~0.41 to 0.67+ over 20 epochs

### Key Improvements Implemented

- ✅ **Distributed Training**: Multi-GPU support with DistributedDataParallel
- ✅ **Enhanced Monitoring**: Comprehensive metrics logging with visual formatting
- ✅ **Memory Efficiency**: Optimized parameters for RTX 3090 hardware
- ✅ **Real-time Features**: Proper time feature extraction and propagation
- ✅ **Model Checkpointing**: Best model saving with DDP compatibility

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

## 🎯 Development Status & Future Improvements

### ✅ Completed Features
- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Real-time Features**: Hour-of-day and day-of-year temporal embeddings
- **Enhanced Monitoring**: Detailed metrics logging with per-horizon breakdown
- **Model Evaluation**: Complete testing framework with baseline comparisons
- **Memory Optimization**: Efficient training on RTX 3090 hardware

### 🚧 In Progress
- [ ] Full 50-epoch training runs for optimal performance
- [ ] Advanced evaluation against more sophisticated baselines

### 🔮 Future Enhancements

#### High Priority
- [ ] Increase input window size to 72-168 timesteps for better temporal context
- [ ] Implement advanced loss functions (Huber, Focal loss)
- [ ] Learning rate scheduling and optimization
- [ ] Model pruning and quantization for deployment

#### Medium Priority
- [ ] Data augmentation strategies for robustness
- [ ] Deeper LLM architectures (6-12 layers)
- [ ] Ensemble methods for improved accuracy
- [ ] Real-time inference optimization

#### Low Priority
- [ ] Advanced attention mechanisms for spatio-temporal fusion
- [ ] Physics-informed constraints and regularization
- [ ] Hyperparameter optimization with Optuna/Ray Tune
- [ ] Multi-modal feature integration (solar wind, magnetic field)

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