import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('url : ', url)

df = pd.read_csv(url, header=None, encoding='utf-8')
print(df.head())

print(df.tail())