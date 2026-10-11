import numpy as np 
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6]) # hours studied 
y = np.array([2.1, 3.9, 5.2, 6.1, 7.8, 9.0]) # test scores
# Initialize parameters for gradient descent    
w = 0  # slope
b = 0  # intercept
learning_rate = 0.01 
num_iterations = 1500
tolerance = 1e-7
previous_loss = np.inf

# Gradient descent loop
for i in range(num_iterations):
    # Compute predictions and gradients using the current parameters
    y_pred = w * x + b
    error = y - y_pred

    # Compute gradients
    dw = (-2/len(x)) * np.sum(x * error) # gradient with respect to w
    db = (-2/len(x)) * np.sum(error) # gradient with respect to b

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Measure loss after the update so it matches the reported parameters
    y_pred = w * x + b
    loss = np.mean((y - y_pred) ** 2)
    if i % 100 == 0:
        print(f"Iteration {i + 1}: Loss = {loss}")

    if abs(previous_loss - loss) < tolerance:
        print(f"Converged after {i + 1} iterations.")
        break
    previous_loss = loss

print(f"Final loss: {loss}")
print(f"Optimized slope (w): {w}")
print(f"Optimized intercept (b): {b}")

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, w * x + b, color='red', label='Fitted line')
plt.xlabel('Hours Studied')
plt.ylabel('Test Scores')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()
