import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. Load dataset (pandas)
# ======================
df = pd.read_csv("Project-Sleep-Quality/DataSet.csv")

# Features & target
X_data = df.drop("sleep_quality", axis=1).values
y = df["sleep_quality"].values

m = len(y)

# ======================
# 2. Normalization (important)
# ======================
X_data = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)

# Add bias
X = np.column_stack((np.ones(m), X_data))

# Initialize theta
theta = np.zeros(X.shape[1])

# Hyperparameters
alpha = 0.01
iterations = 30000

# ======================
# 3. Sigmoid
# ======================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ======================
# 4. Cost function
# ======================
def cost(theta):
    z = X.dot(theta)
    h = sigmoid(z)
    
    epsilon = 1e-5
    return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

# ======================
# 5. Gradient Descent
# ======================
costs = []

for i in range(iterations):
    z = X.dot(theta)
    h = sigmoid(z)
    
    gradient = (X.T @ (h - y)) / m
    theta -= alpha * gradient
    
    costs.append(cost(theta))
    
    if i % 2000 == 0:
        print(f"Iteration {i}, Cost: {cost(theta)}")

# ======================
# 6. Results
# ======================
print("\nFinal theta:", theta)
print("Final cost:", cost(theta))

# ======================
# 7. Prediction
# ======================
def predict(X, theta):
    probs = sigmoid(X.dot(theta))
    return (probs >= 0.5).astype(int)

predictions = predict(X, theta)

accuracy = np.mean(predictions == y) * 100
print("Accuracy:", accuracy, "%")

# ======================
# 8. Plot cost curve
# ======================
plt.figure()
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Training Curve")
plt.show()