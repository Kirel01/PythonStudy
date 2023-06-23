from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = [[1], [2], [3], [4], [5]]
Y = [[2], [5], [8], [13], [14]]

plt.rcParams['font.family'] = 'Malgun Gothic'   #범례를 한글로 표시
plt.rcParams['axes.unicode_minus'] = False
plt.title("선형회귀분석")
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X,Y)
plt.show()

#선형회귀분석
reg = LinearRegression()
Model = reg.fit(X, Y)           # 학습(훈련) 실행

print("============================")
print("기울기 : ", Model.coef_, " 절편 : ", Model.intercept_) # y = ax + b
print("15의 예측치 : ", Model.predict([[15]]))
print("============================")

plt.title("선형회귀분석 결과")
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X,Y)
plt.plot(X, Model.coef_*X+Model.intercept_)     # 선형회귀분석 통해 얻은 선 그래프
plt.show()