import numpy as np 

x = np.array([1, 2, 3, 4, 5, 6]) # hours studied 
y = np.array([2.1, 3.9, 5.2, 6.1, 7.8, 9.0]) # test scores
# Initialize parameters for gradient descent    
w = 0  # slope
b = 0  # intercept
learning_rate = 0.01 
num_iterations = 1500

# Gradient descent loop
for i in range(num_iterations):
    # Compute predictions
    y_pred = w * x + b
    error = y - y_pred
    loss = np.mean(error ** 2)  # Mean squared error
    
    # Compute gradients
    dw = (-2/len(x)) * np.sum(x * error) # gradient with respect to w
    db = (-2/len(x)) * np.sum(error) # gradient with respect to b
    
    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}")

print(f"Optimized slope (w): {w}")
print(f"Optimized intercept (b): {b}")
