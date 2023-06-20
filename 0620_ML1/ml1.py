import pandas as pd

data = [1, 3, 5, 7, 9]

#판다스 1차원 데이터셋
s = pd.Series(data)
print("--------")
print(s)

#판다스 2차원 데이터셋
data = {
    '년도' : [2018, 2020, 2023],
    'GPT' : ['GPT-1', 'GPT-3', 'GPT-4'],
    '모델' : ['Transformer', 'Transformer', 'Transformer' ]
}

df = pd.DataFrame(data)
print("--------")
print(df)

