# Pulsar Star Classification Project 🚀🌌

## Overview
This project aims to classify pulsar stars using SVM model with different kernels and hyperparameters. The primary goal is to achieve high accuracy in distinguishing between pulsar stars and non-pulsar stars based on the given dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Class Distribution](#class-distribution)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering & Scaling](#feature-engineering-&-scaling)
6. [Model Training](#model-training)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Introduction
Pulsar stars are highly magnetized rotating neutron stars that emit beams of electromagnetic radiation. The classification of pulsar stars is crucial for astrophysics research. In this project, we use Support Vector Machine (SVM) model to classify pulsar stars based on their features.

## Dataset Overview 📊
### - The dataset consists of **9 variables**: 
- **8 continuous variables**: `IP Mean`, `IP Sd`, `IP Kurtosis`, `IP Skewness`, `DM-SNR Mean`, `DM-SNR Sd`, `DM-SNR Kurtosis`, `DM-SNR Skewness`
- **1 discrete variable**: `target_class` (the target variable)

## Class Distribution ⚖️
### - Class labels: 
- `0` (not a pulsar): **90.84%**
- `1` (pulsar): **9.16%**  
This significant imbalance will be addressed in the modeling phase.

## Data Preprocessing 🧹
### The dataset used for this project contains features derived from pulsar stars. The data preprocessing steps include:
- Handling missing values
- Normalizing the features
- Splitting the dataset into training and testing sets

## Feature Engineering & Scaling 📈
### Prepared feature vectors and target variable:
- **Features**: All columns except `target_class`
- **Target**: `target_class`
- Split data into training (80%) and testing (20%) sets. 
- Applied **Standard Scaling** to ensure all features contribute equally to the model.

## Model Training 🏋️‍♂️
### The SVM Model was trained with different kernels and hyperparameters to optimize performance. The model with different hyperparameters used include:
- SVC (Support Vector Classifier) with default parameters
- SVC with RBF kernel
- SVC with linear kernel
- SVC with polynomial kernel
- SVC with sigmoid kernel

## Results 📈
### The classification accuracy of the models is summarized below:

- Default SVC Model Accuracy: **0.9827**
- RBF Kernel (C = 100.0): **0.9832**
- RBF Kernel (C = 1000.0): **0.9816**
- Linear Kernel (C = 1.0): **0.9830**
- Linear Kernel (C = 100.0): **0.9832**
- Linear Kernel (C = 1000.0): **0.9832**
- Polynomial Kernel (C = 1.0): **0.9807**
- Polynomial Kernel (C = 100.0): **0.9824**
- Sigmoid Kernel (C = 100.0): **0.8858**

### Confusion Matrix
        
          [[3289 17] 
          [ 44 230]]
  

- True Positive (TP): **3289**
- True Negative (TN): **230**
- False Positive (FP): **17**
- False Negative (FN): **44**

### Classification Metrics
- Classification Accuracy: **0.9830**
- Classification Error: **0.0170**
- Precision Score: **0.9949**
- Recall Score: **0.9868**
- F1-Score: **0.9908**

## Conclusion 🎉
The project successfully demonstrates the effectiveness of SVM models in classifying pulsar stars. The results indicate that certain kernel functions yield higher accuracy, contributing to a better understanding of pulsar star classification.

Here is the link for the Streamlit GUI for Pulsar Classification:
https://pulsar-prediction-system.streamlit.app/
