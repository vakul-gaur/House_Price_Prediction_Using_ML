# 🏠 House Price Prediction using California Housing Dataset

This project demonstrates a machine learning workflow to predict house prices using the California Housing dataset. It includes data preprocessing, model training, evaluation, hyperparameter tuning, and visualization of results.

---

## 📊 Dataset Description

The dataset used is the **California Housing dataset** available from `sklearn.datasets`. It consists of housing data collected from the 1990 California census. The dataset includes:

- **Total Samples**: 20,640
- **Features (8)**:
  - `MedInc` – Median income in block group
  - `HouseAge` – Median house age in block group
  - `AveRooms` – Average number of rooms
  - `AveBedrms` – Average number of bedrooms
  - `Population` – Block group population
  - `AveOccup` – Average house occupancy
  - `Latitude` – Block group latitude
  - `Longitude` – Block group longitude
- **Target**: `PRICE` (Median house value for California districts)

---

## 🔧 Data Preprocessing

The following steps were performed to prepare the data:

1. **Load Dataset**: Using `fetch_california_housing()` from scikit-learn.
2. **Convert to DataFrame**: For easier data handling and analysis.
3. **Check for Missing Values**: Ensured no null or NaN values.
4. **Feature-Target Split**: Split dataset into `X` (features) and `y` (target).
5. **Feature Scaling**: Standardized features using `StandardScaler` for better model performance.
6. **Train-Test Split**: Divided data into 80% training and 20% testing.

---

## 🤖 Models Used

Three regression models were implemented:

- **Linear Regression**
- **Decision Tree Regressor**
- **Ridge Regression**

### 📈 Model Evaluation Metrics:
Each model was evaluated using:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score** – Measures goodness of fit (closer to 1 is better)

### 🔍 Hyperparameter Tuning:
- Tuned `max_depth` for `DecisionTreeRegressor`
- Tuned `alpha` for `Ridge` regression

---

## 📊 Visualization

The project includes the following visual outputs:
- **Predicted vs Actual Scatter Plots** for Linear Regression and Decision Tree
- **Residual Distribution Plots** for both models
- These help visualize model accuracy and error distribution.
