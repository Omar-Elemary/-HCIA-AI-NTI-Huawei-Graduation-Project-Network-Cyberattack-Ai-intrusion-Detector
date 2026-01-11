# Cyber Attack Detection using Deep Learning

A comprehensive deep learning-based Intrusion Detection System (IDS) for identifying various types of cyber attacks in network traffic. This project uses the CICIDS2017 dataset to train a neural network capable of classifying 10 different attack types with high accuracy.

## 🎯 Project Overview

This project implements a trustworthy AI system for cybersecurity that can automatically detect and classify multiple types of cyber attacks in network traffic, including:

- **BENIGN** (Normal traffic)
- **DoS Hulk**
- **PortScan**
- **DDoS** (Distributed Denial of Service)
- **DoS GoldenEye**
- **DoS slowloris**
- **DoS Slowhttptest**
- **Bot**
- **Infiltration**
- **Heartbleed**

## ✨ Key Features

- **Deep Neural Network Architecture**: Multi-layer perceptron with advanced regularization techniques
- **Handles Class Imbalance**: Uses SMOTE (Synthetic Minority Oversampling) for rare attack types
- **Robust Preprocessing**: Feature engineering, correlation analysis, and feature selection
- **High Performance**: Achieves **97.27% accuracy** on test data
- **Model Calibration**: Temperature scaling for confidence calibration
- **Production Ready**: Includes deployment-ready inference code
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and overfitting detection

## 📊 Dataset

This project uses the **CICIDS2017** (Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017), which contains:
- **2,214,469** network flow samples
- **79** original features
- **10** distinct classes (1 benign + 9 attack types)

### Class Distribution
- BENIGN: 75.54%
- DoS Hulk: 10.43%
- PortScan: 7.18%
- DDoS: 5.78%
- Other attack types: < 1% each

## 🏗️ Architecture

### Model Architecture
- **Input Layer**: 40 selected features
- **Hidden Layers**:
  - Dense(512) → LeakyReLU → BatchNorm → Dropout(0.4)
  - Dense(256) → LeakyReLU → BatchNorm → Dropout(0.4)
  - Dense(128) → LeakyReLU → BatchNorm → Dropout(0.3)
  - Dense(64) → LeakyReLU → BatchNorm
- **Output Layer**: Dense(10) with Softmax activation

### Key Techniques
- **Focal Loss**: Handles class imbalance effectively
- **L2 Regularization**: Prevents overfitting
- **Batch Normalization**: Stabilizes training
- **Dropout**: Regularization for better generalization
- **RobustScaler**: Handles outliers in feature scaling

## 📈 Performance Metrics

### Overall Performance (Test Set)
- **Accuracy**: 97.27%
- **Precision**: 97.36% (weighted)
- **Recall**: 97.27% (weighted)
- **F1-Score**: 97.13% (weighted)

### Per-Class Performance
- **BENIGN**: Precision 96.70%, Recall 99.85%
- **DDoS**: Precision 99.90%, Recall 99.87%
- **PortScan**: Precision 99.43%, Recall 99.87%
- **DoS GoldenEye**: Precision 98.90%, Recall 96.11%
- Other classes show varying performance based on class frequency

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow joblib kagglehub
```

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd "Final project"
```

2. The dataset will be automatically downloaded from Kaggle when you run the notebook (requires Kaggle credentials):
```python
import kagglehub
path = kagglehub.dataset_download("sweety18/cicids2017-full-dataset")
```

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook "final_cyber_attack_detection_fixed (3).ipynb"
```

2. Run all cells sequentially. The notebook includes:
   - Data loading and exploration
   - Data preprocessing and cleaning
   - Feature engineering and selection
   - Model training
   - Evaluation and visualization
   - Model saving

### Model Files

After training, the following files will be generated:
- `best_ids_model.keras` - Trained neural network model
- `robust_scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder
- `feature_names.json` - Selected feature names
- `model_temperature.json` - Temperature calibration parameter
- `training_history.png` - Training curves visualization
- `confusion_matrix.png` - Confusion matrix visualization

## 💻 Usage

### Production Inference

The notebook includes a `CyberAttackDetector` class for production use:

```python
from CyberAttackDetector import CyberAttackDetector

# Initialize detector
detector = CyberAttackDetector()

# Prepare network packet features (DataFrame or numpy array)
# Features must match the 40 selected features

# Predict attack type
result = detector.predict(network_packet)

print(f"Attack Type: {result['attack_type']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Is Attack: {result['is_attack']}")
print(f"All Probabilities: {result['all_probabilities']}")
```

## 🔬 Methodology

### Data Preprocessing Pipeline

1. **Data Cleaning**:
   - Remove infinite values and NaN
   - Drop zero-variance columns
   - Handle duplicates (after train/test split)

2. **Feature Engineering**:
   - Correlation analysis (removed features with >95% correlation)
   - Derived features (bytes per packet, log transforms, etc.)
   - Feature selection using Random Forest importance (top 40 features)

3. **Data Transformation**:
   - Log1p transformation for numerical stability
   - RobustScaler for feature scaling
   - One-hot encoding for labels

4. **Handling Imbalance**:
   - SMOTE oversampling for minority classes
   - Minimum sample threshold of 5,000 per class

### Training Strategy

- **Train/Test Split**: 80/20 with stratification
- **Train/Validation Split**: 80/20 from training set
- **Batch Size**: 1,048
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Focal Loss (γ=2.0, α=0.5)
- **Callbacks**: Early Stopping, ReduceLROnPlateau
- **Epochs**: 5 (with early stopping)

## 📋 Project Structure

```
Final project/
│
├── final_cyber_attack_detection_fixed (3).ipynb  # Main notebook
├── Trustworthy_AI_for_Cybersecurity (1).pdf      # Project documentation
├── README.md                                     # This file
│
├── best_ids_model.keras                          # Trained model (generated)
├── robust_scaler.pkl                             # Scaler (generated)
├── label_encoder.pkl                             # Label encoder (generated)
├── feature_names.json                            # Feature names (generated)
├── model_temperature.json                        # Temperature (generated)
├── training_history.png                          # Training curves (generated)
└── confusion_matrix.png                          # Confusion matrix (generated)
```

## 🔍 Key Insights

1. **Feature Importance**: Packet length statistics (max, mean, variance) are most discriminative
2. **Class Imbalance**: Rare classes (Infiltration, Heartbleed) remain challenging even after SMOTE
3. **Model Calibration**: Temperature scaling shows the model is well-calibrated
4. **Overfitting**: Model shows minimal overfitting (gap < 2% between train/val accuracy)

## ⚠️ Limitations

- Rare attack classes (Infiltration, Heartbleed) have limited samples, affecting performance
- Model trained on CICIDS2017 may need retraining for different network environments
- Real-world deployment would require integration with network monitoring tools

## 🔮 Future Improvements

- Real-time streaming detection
- Online learning for new attack patterns
- Ensemble methods for improved robustness
- Explainability features (SHAP, LIME)
- Integration with SIEM systems
- Anomaly detection for zero-day attacks

## 📚 References

- CICIDS2017 Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/sweety18/cicids2017-full-dataset)
- Canadian Institute for Cybersecurity: [CIC Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

## 📝 License

This project is part of academic coursework. Please refer to the dataset license for CICIDS2017 usage terms.

## 👥 Authors

ETA AI Final Project - Trustworthy AI for Cybersecurity

## 🙏 Acknowledgments

- Canadian Institute for Cybersecurity for the CICIDS2017 dataset
- Kaggle community for dataset hosting

