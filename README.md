# Time Series Missing Value Imputation Project

## Overview

This project implements and compares multiple missing value imputation methods for time series data, specifically focusing on air quality (O₃) sensor data from multiple locations in Barcelona. The project evaluates the performance of different imputation techniques using both univariate and multivariate approaches.

## Dataset

The dataset contains hourly O₃ (ozone) concentration measurements from 8 air quality monitoring stations in Barcelona:
- **gracia**: Gràcia district
- **pr**: Poblenou district  
- **eixample**: Eixample district
- **prat**: El Prat de Llobregat
- **montcada**: Montcada i Reixac
- **ciutadella**: Parc de la Ciutadella
- **hebron**: Carrer de l'Hèbron
- **badalona**: Badalona

**Data characteristics:**
- Time period: June 2017 (hourly measurements)
- Total records: 2,257 time points
- Missing data: 20% artificially introduced with burst patterns (5 consecutive missing values)

## Methods Implemented

### 1. Univariate Methods

#### Polynomial Interpolation
- **Method**: 3rd degree polynomial interpolation
- **Approach**: Uses temporal patterns within individual sensors
- **Performance**: RMSE: 16.27, R²: 0.51

#### LSTM-based Imputation
- **Architecture**: 2-layer LSTM with 128 hidden units
- **Sequence length**: 6 time steps
- **Training**: 30 epochs with Adam optimizer
- **Performance**: RMSE: 17.24, R²: 0.45

### 2. Multivariate Methods

#### MICE (Multiple Imputation by Chained Equations)
- **Linear Regression variant**: Uses Bayesian Ridge regression
- **KNN variant**: Uses K-Nearest Neighbors (k=5)
- **Iterations**: 10 maximum iterations
- **Performance (Linear)**: RMSE: 9.78-16.74, R²: 0.56-0.89
- **Performance (KNN)**: RMSE: 10.43-15.99, R²: 0.59-0.87

#### Autoencoder
- **Architecture**: 8-8-2-8-8 (encoder-decoder)
- **Training**: 100 epochs with iterative imputation
- **Loss**: MSE with missing value masking
- **Performance**: RMSE: 12.68-16.96, R²: 0.55-0.77

## Project Structure

```
├── main.ipynb                # Main Jupyter notebook with all experiments
├── data_matrix.csv           # Original air quality dataset
├── missing_mask.csv          # Generated missing value mask
├── missing_generator.py      # Utility for creating missing value patterns
├── README.md                 # This file
├── TOML_Report.pdf           # Report of the project
└── requirements.txt          # Python dependencies
```

## Key Features

### Missing Value Generation
- **Bursty missing patterns**: Simulates realistic sensor failures
- **Configurable parameters**: Missing percentage and burst length
- **Reproducible**: Uses fixed random seed for consistency

### Evaluation Framework
- **Metrics**: RMSE and R² score
- **Validation**: Only evaluates on artificially introduced missing values
- **Visualization**: Zoomed-in plots around missing data regions

### Performance Comparison
| Method | Best RMSE | Best R² | Sensor |
|--------|-----------|---------|---------|
| MICE Linear | 9.78 | 0.89 | gracia |
| MICE KNN | 10.43 | 0.87 | montcada |
| Autoencoder | 12.68 | 0.77 | hebron |
| Polynomial | 16.27 | 0.51 | gracia |
| LSTM | 17.24 | 0.45 | gracia |

## Installation and Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Virtual environment (recommended)

### Dependencies
```bash
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install torch
pip install "numpy<2"
```

### Running the Project
1. Clone the repository
2. Install dependencies
3. Open `main.ipynb` in Jupyter
4. Run cells sequentially

## Results and Insights

### Key Findings
1. **MICE methods perform best**: Both linear and KNN variants achieve the lowest RMSE and highest R² scores
2. **Multivariate approaches superior**: Leveraging correlations between sensors improves imputation quality
3. **LSTM underperforms**: May need more data or different architecture for this specific task
4. **Polynomial interpolation**: Simple but effective for univariate scenarios

### Performance by Sensor
- **Best performing sensors**: gracia, montcada, hebron
- **Challenging sensors**: ciutadella (lower R² scores)
- **Consistent patterns**: MICE methods show consistent performance across sensors

## Technical Details

### Data Preprocessing
- **Normalization**: MinMax scaling for neural network methods
- **Sequence preparation**: Sliding window approach for LSTM
- **Missing value handling**: Mean imputation for autoencoder initialization

### Model Architectures
- **LSTM**: 2 layers, 128 hidden units, sequence length 6
- **Autoencoder**: 8-8-2-8-8 architecture with ReLU activation
- **MICE**: Bayesian Ridge and KNN estimators

### Training Parameters
- **LSTM**: 30 epochs, batch size 32, learning rate 0.001
- **Autoencoder**: 100 epochs, learning rate 0.01
- **MICE**: 10 iterations, random state 42
