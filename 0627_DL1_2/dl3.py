#붓꽃 데이터 그래프

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('url : ', url)

df = pd.read_csv(url, header=None, encoding='utf-8')

'''
    -- Iris Setosa  (0)
    -- Iris Versicolour
    -- Iris Virginica (0)
'''
y = df.iloc[0:100, 4].values
# np.where(y='Iris-setosa', )

# 꽃밭침 길이와 꽃잎 길이를 추출
X = df.iloc[0:100, [0, 2]].values

# 산점도 그림
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label="Versicolour")

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()