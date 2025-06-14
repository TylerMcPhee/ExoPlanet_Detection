# Exoplanet Detection with 1D CNNs

This project implements a complete deep learning pipeline to detect exoplanets using time-series data from the Kepler space telescope. It includes preprocessing of raw data, signal smoothing, frequency domain transformation via FFT, handling of class imbalance, and training a 1D Convolutional Neural Network (CNN) to perform binary classification.

## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Project Structure](#project-structure)  
- [Preprocessing Pipeline](#preprocessing-pipeline)  
- [Model Architecture](#model-architecture)  
- [Training and Evaluation](#training-and-evaluation)  
- [Results](#results)  
- [Acknowledgements](#acknowledgements)

## Overview

Exoplanets are planets that orbit stars outside our solar system. Detecting them involves analyzing subtle, periodic dips in a star’s brightness. This project leverages signal processing techniques and a 1D CNN architecture to identify such patterns from light curves recorded by the Kepler telescope.

## Dataset

The dataset used is the **Kepler Labeled Time Series Dataset**, available on [Kaggle](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data).

- **exoTrain.csv**: Contains 5,087 labeled samples.
- Each sample includes time-series readings of stellar brightness.
- Labels: `1` (confirmed exoplanet), `0` (no exoplanet).

## Preprocessing Pipeline
1. Data Loading: Reads compressed CSV file directly from .zip.

2. Feature Selection: Extracts time-series data and target labels.

3. Normalization: Applies L2 normalization to each time-series.

4. Smoothing: Uses a Gaussian filter to reduce signal noise.

5. FFT Transformation: Converts time-series data to frequency domain.

6. Oversampling: Balances the dataset using RandomOverSampler to improve minority class performance.

7. Reshaping: Reshapes input into (samples, sequence_length, 1) format required for 1D CNNs.

## Model Architecture
- The model is a Fully Convolutional Network (FCN) implemented in Keras/TensorFlow:

- Conv1D (256 filters, kernel size 8) → MaxPooling → BatchNormalization

- Conv1D (340 filters, kernel size 6) → MaxPooling → BatchNormalization

- Conv1D (256 filters, kernel size 4) → MaxPooling → BatchNormalization

- Flatten → Dropout (0.3)

- Dense (24) → Dropout (0.3)

- Dense (12) → Dense (8) → Dense (1, sigmoid)

Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 10
Batch Size: 10

## Training and Evaluation
- Training is performed on the oversampled frequency-domain data.

- Evaluation is conducted on the held-out test set.

- Plots for training accuracy and loss are generated using matplotlib.

## Results
- Achieved high training accuracy (>95%) in just 10 epochs.

- Oversampling significantly improved model sensitivity to the minority class.

- The FFT + CNN pipeline effectively captured periodic patterns associated with exoplanet transits.
