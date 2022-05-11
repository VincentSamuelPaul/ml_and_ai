import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')


def gradient_descent(m, b, data, L):
    m_gradient = 0
    b_gradient = 0
    n = len(data['0'])
    for i in range(n):
        x = data['0'][i]
        y = data['1'][i]
        m_gradient += -(2/n) * x * (y - (m*x + b))
        b_gradient += -(2/n) * (y - (m*x + b))
    m = m - L * m_gradient
    b = m - L * b_gradient
    return m, b

m = 0
b = 0
L = 0.0001
iters = 5000

for i in range(iters):
    if i % 100 == 0:
        print(f'Iters : {i}')
    m, b = gradient_descent(m, b, data, L)
print(f'm: {m}, \nb: {b}')

# predicting miles per amt

amount = int(input('How is the amount: '))

pred = m * amount + b
print(pred)



plt.scatter(data['0'], data['1'], c='orange')
plt.plot( list(range(30, 45)), [m*x + b for x in range(30,45)], c='red' )
plt.xlabel('Amount')
plt.ylabel('No of Miles')
plt.show()