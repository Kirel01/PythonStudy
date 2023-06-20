import pandas as pd

sample = pd.read_csv('sample.csv')
print(sample)
print(sample.info())
print(sample.describe())