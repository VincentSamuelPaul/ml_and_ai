import pandas as pd

train = pd.read_csv('knn.data')


cols = ['Gender', 'Age', 'Salary', 'Purchase_Iphone']
train.rename(columns={cols[0]:0, cols[1]:1, cols[2]:2}, inplace=True)
train['distance'] = 1

target = pd.Series(['Male', 40, 75000])
train['distance'] = ((train.loc[:,1]-target[1])**2 + (train.loc[:,2]-target[2])**2) ** 0.5

k = 7
train = train.sort_values('distance', ascending=True)
knn = list(train.head(k).Purchase_Iphone)

print(train.head())
print(knn)

from statistics import mode

print(mode(knn))

import matplotlib.pyplot as plt

colors = {1:'red', 0:'blue'}
plt.scatter(
    train[1],
    train[2],
    c=train['Purchase_Iphone'].map(colors)
)
plt.scatter(
    target[1],
    target[2],
    c='orange'
)
plt.xlabel(cols[1])
plt.ylabel(cols[2])
plt.title('Predict Buying Iphone')
plt.show()
