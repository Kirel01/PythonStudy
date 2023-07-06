import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


#1번

test_data = {
    'name' : ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'horse power' : [130, 250, 190, 300, 210, 220, 170],
    'efficiency' : [16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2]
}
data = DataFrame(test_data)
# 'name'을 index로 설정
data.set_index('name', inplace=True)
print(data)


# 2번

X = data[['horse power']]
Y = data['efficiency']

regression = LinearRegression() # LinearRegression 모델 인스턴스 생성
regression.fit(X, Y) # 데이터를 통해 모델 훈련

# 계수, 절편 및 예측 점수를 출력
print("계수:", regression.coef_)
print("절편:", regression.intercept_)
score = regression.score(X, Y)
print("예측 점수:", score)


# 3번

car = [[280]]
car_efficiency = regression.predict(car)
# 소수점 두 번째 자리까지 반올림
car_efficiency_round = round(car_efficiency[0], 2)
print("280 마력 자동차의 예상 연비",car_efficiency_round, "km/l")