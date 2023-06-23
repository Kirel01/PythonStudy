# iris 꽃 데이터 로드
from sklearn.datasets import load_iris

# load_iris 데이터 셋 불러옴
iris = load_iris()

# feature 데이터 값
X = iris['data']
#print(X[:])
print(X[:5])    # 5개만 출력
print("==============================")

# target 데이터 값
Y = iris['target']
#print(Y[:])
print(Y[:5])    # 5개만 출력
print("==============================")

# 이 데이터가 어떤것인지 설명
print(iris['DESCR'])

print("==============================")

# 데이터의 각 항목
print(iris['feature_names'])

print("==============================")

# 데이터에 대한 타켓(결과)
print(iris["target_names"])