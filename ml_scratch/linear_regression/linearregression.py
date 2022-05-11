import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('weatherHistory.csv')
data = data.head(100)

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].experience
        y = points.iloc[i].salary
        total_error += (y - (m*x + b)) ** 2
    total_error / float(len(points))

def gradient_descent(m_now, b_now, data, L):
    m_gradient = 0
    b_gradient = 0

    n = len(data['Temperature'])

    for i in range(n):
        x = data.iloc[i].Temperature
        y = data.iloc[i].Humidity

        m_gradient += -(2/n) * x * (y - (m_now*x + b_now))
        b_gradient += -(2/n) * (y - (m_now*x + b_now))

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return m, b


m = 0
b = 0
L = 0.0001
epochs = 5000

for i in range(epochs):
    if i % 50 == 0:
        print(f'Epoch : {i}')
    m, b = gradient_descent(m, b, data, L)

print(m,b)

# predicting Humidity

salary_pred = m * 2 + b
print(salary_pred)

plt.scatter(data['Temperature'], data['Humidity'], c='orange')
plt.plot(list(range(0,25)), [m*x + b for x in range(0,25)], c='red')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()

#0.036154797644272936, 0.10744235052230534

