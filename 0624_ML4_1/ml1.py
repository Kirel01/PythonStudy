# 1. 데이터 및 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
X = fetch_california_housing(as_frame=True)['data']
y = fetch_california_housing(as_frame=True)['target']

print(X)
print(y)

'''
MedInc : 소득(중앙값)
HouseAge : 주택 나이(중앙값)
AveRooms : 전체 방 수
AveBedrs : 전체 침실수
Pupulation : 인구
AveOccup : 가족 구성원 수
Latitude : 위도
Longitude : 경도
'''

data = pd.concat([X,y], axis=1)
print(data)
print(data.info())

#2. 데이터 수집 및 정제 -- 결측치 없음
print(data.info())

#3. EDA
print(data.head())