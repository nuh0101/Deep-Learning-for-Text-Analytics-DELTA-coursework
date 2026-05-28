**Project Overview**
This project focuses on predicting a continuous target variable from a tabular dataset (19 numerical features) using deep learning. The goal was to build, evaluate, and iteratively optimize a neural network in TensorFlow/Keras to outperform traditional machine learning baselines.

**Methodology & Workflow**
Data Preprocessing: Handled data ingestion, performed an 80/20 train/validation split, and applied standard scaling (StandardScaler) to normalize feature distributions for stable neural network training.

Benchmarking: Established strong performance baselines using traditional algorithms:

Linear Regression (RMSE: ~86.9)

Random Forest Regressor (RMSE: ~52.8)

Neural Network Optimization: Started with a basic, untuned multi-layer perceptron and iteratively improved the architecture using KerasTuner.

Key Breakthroughs for Tabular Data: * Discovered that deep architectures with heavy dropout underperformed on this specific dataset.

Pivoted to a shallower, wider architecture (2 hidden layers with 192 and 128 neurons).

Integrated Batch Normalization to stabilize learning and used Early Stopping (patience=25) to prevent overfitting.

Increased the batch size to 64, which significantly smoothed the gradient updates and eliminated validation noise.

**Results**
The iterative tuning process was highly successful. The final optimized Keras model achieved a Validation RMSE of 19.42, cutting the error rate of the Random Forest baseline by more than half. The model demonstrated excellent generalization, evidenced by a perfectly converging train/validation loss curve.

Tech Stack
Language: Python
Data Manipulation: Pandas, NumPy
Machine Learning: Scikit-Learn (Preprocessing, Random Forest, Linear Models)
Deep Learning: TensorFlow, Keras, KerasTuner
Visualization: Matplotlib
