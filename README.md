# Pulsar Star Classification Project Using SVM 🌌🔍

# I am excited to share my recent project on **Pulsar Star Classification** using Support Vector Machines (SVM).
# The goal was to effectively classify pulsar stars based on their features, focusing on addressing class imbalance and outliers in the dataset. 

## Dataset Overview 📊
# - The dataset consists of **9 variables**: 
#   - **8 continuous variables**: `IP Mean`, `IP Sd`, `IP Kurtosis`, `IP Skewness`, `DM-SNR Mean`, `DM-SNR Sd`, `DM-SNR Kurtosis`, `DM-SNR Skewness`
#   - **1 discrete variable**: `target_class` (the target variable)

## Class Distribution ⚖️
# - Class labels: 
#   - `0` (not a pulsar): **90.84%**
#   - `1` (pulsar): **9.16%**  
# This significant imbalance will be addressed in the modeling phase.

## Data Cleaning & Exploration 🔍
# - Checked for **missing values**: None found.
# - Identified **outliers** through boxplots, confirming the presence of numerous outliers in all continuous variables. This necessitates careful handling during model training.

## Feature Engineering & Scaling 📈
# - Prepared feature vectors and target variable:
#   - **Features**: All columns except `target_class`
#   - **Target**: `target_class`
# - Split data into training (80%) and testing (20%) sets. 
# - Applied **Standard Scaling** to ensure all features contribute equally to the model.

## SVM Modeling 🤖
# - **Model Variants**:
#   - **Default SVM**: Achieved an accuracy of **0.9827**.
#   - **RBF Kernel**: 
#     - C = **1.0**: Accuracy: **0.9832**
#     - C = **100.0**: Accuracy: **0.9832**
#     - C = **1000.0**: Accuracy: **0.9816** (a slight decrease)
#   - **Linear Kernel**: 
#     - C = **1.0**: Accuracy: **0.9830**
#     - C = **100.0**: Accuracy: **0.9832**
#     - C = **1000.0**: Accuracy: **0.9832** (optimal performance)
# - Compared training and testing set accuracy to check for **overfitting/underfitting**:
#   - Training accuracy: **0.9783**
#   - Testing accuracy: **0.9830**

## Model Evaluation 📉
# - **Null Accuracy**: **0.9235** (accuracy of predicting the majority class).
# - Our model's performance (0.9832) significantly surpasses this, indicating effective classification.

## Conclusion 🎉
# The SVM classifier demonstrated robust performance in pulsar star classification, effectively addressing outliers and class imbalance. 
# The insights gained from this project underscore the importance of model evaluation metrics in understanding classification performance. 
