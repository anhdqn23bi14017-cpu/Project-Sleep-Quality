import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1.Load dataset
df = pd.read_csv("Project-Sleep-Quality/DataSet.csv")
X_data = df.drop("sleep_quality", axis=1).values
y = df["sleep_quality"].values

#2.Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.2, random_state=42, stratify=y
)

#3.Normalise using TRAINING set statistics
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train_norm = (X_train - mean) / std
X_test_norm  = (X_test - mean) / std

#Add bias column
m_train = len(y_train)
m_test  = len(y_test)

X_train = np.column_stack((np.ones(m_train), X_train_norm))
X_test  = np.column_stack((np.ones(m_test), X_test_norm))

#4.Sigmoid, cost, gradient (with optional L2 regularisation)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, lambda_reg=0.0):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    #Binary cross-entropy
    loss = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    #L2 regularisation (skip bias term theta[0])
    reg = (lambda_reg / (2 * m)) * np.sum(theta[1:] ** 2)
    return loss + reg

def gradient(theta, X, y, lambda_reg=0.0):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    #Regularisation gradient (bias not regularised)
    if lambda_reg > 0:
        grad[1:] += (lambda_reg / m) * theta[1:]
    return grad

#5.Gradient descent with early stopping
def gradient_descent(X, y, alpha=0.1, iterations=10000, lambda_reg=0.0, tol=1e-6):
    theta = np.zeros(X.shape[1])
    costs = []
    prev_cost = float('inf')
    
    for i in range(iterations):
        grad = gradient(theta, X, y, lambda_reg)
        theta -= alpha * grad
        
        #Compute cost every 100 iterations for monitoring
        if i % 100 == 0:
            curr_cost = cost(theta, X, y, lambda_reg)
            costs.append(curr_cost)
            
            #Early stopping
            if abs(prev_cost - curr_cost) < tol:
                print(f"Early stopping at iteration {i} (cost change < {tol})")
                break
            prev_cost = curr_cost
            
        if i % 2000 == 0:
            print(f"Iteration {i}, Cost: {cost(theta, X, y, lambda_reg):.6f}")
    
    return theta, costs

#6.Train the model
alpha = 0.1          #increased learning rate (works well with scaled data)
iterations = 10000
lambda_reg = 0.01    #small L2 regularisation (optional)

theta, costs = gradient_descent(X_train, y_train, alpha, iterations, lambda_reg)

print("\nFinal theta:", theta)
print("Final training cost:", cost(theta, X_train, y_train, lambda_reg))

#7.Predictions & evaluation
def predict(X, theta):
    return (sigmoid(X @ theta) >= 0.5).astype(int)

#Training accuracy (for info only)
train_pred = predict(X_train, theta)
train_acc = np.mean(train_pred == y_train) * 100
print(f"Training accuracy: {train_acc:.2f}%")

#Test accuracy (true generalisation)
test_pred = predict(X_test, theta)
test_acc = np.mean(test_pred == y_test) * 100
print(f"Test accuracy: {test_acc:.2f}%")

#8. Plot cost curve
plt.figure(figsize=(8,5))
plt.plot(costs)
plt.xlabel("Iterations (per 100)")
plt.ylabel("Cost")
plt.title("Training Cost Curve (with early stopping)")
plt.grid(True)
plt.show()
