import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = r'C:\Users\mwend\Downloads\Nairobi Office Price Ex.csv'
data = pd.read_csv(file_path)

x = data['SIZE'].values
y = data['PRICE'].values

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, m, c, learning_rate, num_iterations):
    n = len(y)
    for num_iteration in range(num_iterations):
        y_pred = m * x + c
        error = mean_squared_error(y, y_pred)
        print(f"Iteration {num_iteration+1}: MSE = {error}")

        #to calculate gradients
        dm = (-2/n) * np.sum(x * (y - y_pred))
        dc = (-2/n) * np.sum(y -y_pred)
        m = m -learning_rate * dm
        c = c - learning_rate * dc

    return m, c

m = np.random.rand()
c = np.random.rand()
learning_rate = 0.0001
num_iterations = 10
m, c = gradient_descent(x, y, m, c, learning_rate, num_iterations)

plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, m * x + c, color='red', label='Line of best fit')
plt.xlabel('Office size (sq.ft)')
plt.ylabel('Office price ($)')
plt.title('Linear Regression')
plt.legend()
plt.show()

size = 100
predicted_price = m * size + c
print(f'Predicted price for 100 sq.ft is: ${predicted_price:.2f}')

