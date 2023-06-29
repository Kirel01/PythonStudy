'''
    * 활성화 함수 (activation function)
        - 입력 신호가 출력 결과에 미치는 영향도를 조절하는 매개변수
        - 종류

'''
import numpy as np
import matplotlib.pyplot as plt

def step(x):
    result = x > 0.00000001         # 부동소수점 오차 방지
    return result.astype(int)       # 정수로 변환

x = np.arange(-10.0, 10.0, 0.1)
y = step(x)
plt.plot(x, y)
#plt.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

x = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
#plt.show()

# 지정된 구간내에서 균열한 간격으로 수를 가지는 배열을 반환함
x = np.linspace(-np.pi, np.pi, 60)
#print(x)
y = np.tanh(x)
plt.plot(x, y)
#plt.show()


def relu(x):
    return np.maximum(x, 0)

x = np.arange(-10.0, 10.0, 0.1)
#print(x)
y = relu(x)
plt.plot(x, y)
plt.show()