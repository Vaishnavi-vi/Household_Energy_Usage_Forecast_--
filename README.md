# 🔋 Household Energy Consumption Forecasting

This project aims to predict **Global Active Power (kilowatts)** consumed in a household using various machine learning models. The dataset used is the **Household Power Consumption Dataset**.

---

## 📂 Dataset Overview

- **Source**: UCI Machine Learning Repository  
- **Rows**: ~2 million  
- **Target Variable**: `Global_active_power`  
- **Features**:
  - `Global_reactive_power`
  - `Voltage`
  - `Global_intensity`
  - `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`
---

## 🧹 Data Preprocessing

- Missing values handling
- Feature engineering (timestamp extraction, interaction features)
- Normalization / Scaling (StandardScaler)
- Train-test split (typically 80/20)

---

## 🔧 Models Implemented

### 1. 📈 Linear Regression
- Simple baseline model
- Assumes linear relationship between input features and target
- Easy to interpret, but prone to underfitting

### 2. 🌳 Decision Tree Regressor
- Captures nonlinear relationships
- May overfit on training data
- Visualized using `plot_tree()`

### 3. 🌲 Random Forest Regressor
- Ensemble of multiple decision trees
- Reduces overfitting and improves generalization
- Hyperparameter tuning: `n_estimators`, `n_jobs`, etc.

### 4. 🚀 Gradient Boosting Regressor
- Boosts weak learners sequentially
- Slower than Random Forest but potentially more accurate
- Hyperparameters tuned via GridSearchCV

### 5. ⚡ XGBoost Regressor
- Extreme Gradient Boosting (high-performance)
- Handles missing values internally
- Feature importance visualization using `xgboost.plot_importance()`

### 6. 🧠 Artificial Neural Network (ANN)
- Deep learning model using TensorFlow/Keras
- Architecture:
  - Input layer with 14 features
  - Hidden layers with ReLU activations
  - Output layer with 1 neuron (linear activation)
- Optimizer: Adam | Loss: MSE | Metric: MAE

---

## 📊 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

| Model              | MAE   | MSE   | R² Score |
|-------------------|-------|-------|----------|
| Linear Regression | 0.104 | 0.065 | 0.938    |
| Decision Tree     | 0.031 | 0.022 | 0.980    |
| Random Forest     | 0.025 | 0.001 | 0.998    |
| Gradient Boosting | 0.051 | 0.031 | 0.972    |
| XGBoost           | 0.041 | 0.021 | 0.981    |
| ANN               | 0.138 | 0.097 | 0.912    |

---

## 🖼️ Visualizations

- Correlation Heatmaps
- Feature Importance Plots
- Model Prediction vs Actual Line Charts
- Residual Plots

---

## 📦 Dependencies

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow / keras
