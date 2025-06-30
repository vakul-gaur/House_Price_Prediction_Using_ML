# -----------------------------------------------
# Step 1: Import Libraries & Load Dataset, This section loads necessary libraries, loads the dataset, 
# and prepares it in a structured format for further analysis.
# -----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn modules for data, models, and evaluation
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the California housing dataset (built-in in sklearn)
housing = fetch_california_housing()

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target  # Add target column for prediction

# -----------------------------------------------
# Step 2: Data Preprocessing, Data preprocessing ensures the model gets clean, consistent, and scaled inputs. 
# Standardization is crucial for linear models to converge correctly.
# -----------------------------------------------

# Check for any missing values in the dataset
print("Missing values:\n", df.isnull().sum())

# Separate features (X) and target (y)
X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# Standardize features (important for algorithms like linear and ridge regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# Step 3: Model Selection and Training, Training two different models helps compare simple (linear) and 
# complex (non-linear) approaches for predicting house prices.
# -----------------------------------------------
# Initialize models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)

# Train models on the training data
linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# -----------------------------------------------
# Step 4: Evaluation Function, Evaluation metrics help quantify how good the model is. Comparing different models using 
# the same metrics gives clarity on which one performs better.
# -----------------------------------------------
# Define a function to evaluate model performance
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("MAE:", mean_absolute_error(y_true, y_pred))  # Measures average error
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))  # Penalizes large errors
    print("R² Score:", r2_score(y_true, y_pred))  # Shows how well the model explains variability

# Make predictions using both models
y_pred_linear = linear_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

# Evaluate both models
evaluate_model("Linear Regression", y_test, y_pred_linear)
evaluate_model("Decision Tree", y_test, y_pred_tree)

# -----------------------------------------------
# Step 5: Basic Hyperparameter Tuning, Tuning hyperparameters like max_depth and alpha helps find the best settings for 
# model performance, balancing bias and variance.
# -----------------------------------------------

# Test different values of max_depth for the decision tree
print("\nTuning Decision Tree - max_depth:")
for depth in [2, 4, 6, 8, 10, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    print(f"max_depth={depth} | R²: {r2_score(y_test, pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")

# Test different alpha values for Ridge Regression
print("\nTuning Ridge Regression - alpha:")
for alpha in [0.01, 0.1, 1, 10, 100]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    print(f"alpha={alpha} | R²: {r2_score(y_test, pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")

# -----------------------------------------------
# Step 6: Visualization, Visualizing predicted vs actual prices gives a clear view of model accuracy. 
# The closer the points are to the red line, the better the prediction.
# -----------------------------------------------

# Plot predicted vs actual values for both models
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_linear, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line
plt.title("Linear Regression: Predicted vs Actual")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_tree, color='seagreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line
plt.title("Decision Tree: Predicted vs Actual")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.tight_layout()
plt.show()

# Residual distribution plots to analyze prediction errors, Residual plots help detect bias in the model. 
# Ideally, residuals should be symmetrically distributed around zero. Any patterns suggest model issues.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(y_test - y_pred_linear, kde=True, color='royalblue')
plt.axvline(0, color='red', linestyle='--')
plt.title("Linear Regression Residuals")

plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred_tree, kde=True, color='seagreen')
plt.axvline(0, color='red', linestyle='--')
plt.title("Decision Tree Residuals")

plt.tight_layout()
plt.show()