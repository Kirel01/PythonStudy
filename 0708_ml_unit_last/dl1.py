import numpy as np
import matplotlib.pyplot as plt

def stepfunc(x):
    return np.where(x <= 0, -1, 1)   # x 값이 (x <= 0)을 만족하면 -1 반환, 만족하지 못하면 1 반환

def sigmoidfunc(x):
    return 1 / (1+ np.exp(-x))      # numpy의 exp(x) 함수는 e^x를 구하는 지수 함수이다.

def tanhfunc(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))      # 쌍곡탄젠트 함수

def relufunc(x):
    return np.maximum(0,x)      # 인수로 주어진 수 중 가장 큰 수를 반환

def softplusfunc(x):
    return np.log(1 + np.exp(x))    # softplus 함수

x = np.arange(-6, 6, 0.01)      # x 값으로 -6 ~ 5.9까지 0.01 간격의 소수로 이루어진 배열
y_step = stepfunc(x)
y_sigmoid = sigmoidfunc(x)
y_tanh = tanhfunc(x)
y_relu = relufunc(x)
y_soft = softplusfunc(x)

# 그래프 영역을 1x4로 나눔 (1행 4열)
fig, ax = plt.subplots(1, 4, figsize=(30,3))    # 1행 4열로 한번에 그레프 출력, 사이즈 설정

# 첫 번째 그래프 - step 함수
ax[0].plot(x, y_step, color='blue')     # 선의 색상을 파란색으로 변경
ax[0].set_title('step')   # 함수 이름 설정
ax[0].set_yticks(np.arange(-1, 1.5, 0.5))     # y축의 눈금 위치를 -1, -0.5, 0, 0.5, 1로 설정
ax[0].grid(True)      # 격자 표시를 활성화

# 두 번째 그래프 - logistic sigmoid 함수
ax[1].plot(x, y_sigmoid, color='blue')  # 선의 색상을 파란색으로 변경
ax[1].set_title('logistic sigmoid')     # 함수 이름 설정
ax[1].grid(True)  # 격자 표시를 활성화

# 세 번째 그래프 - tanh(hyperboilc tangent) 함수
ax[2].plot(x, y_tanh, color='blue')     # 선의 색상을 파란색으로 변경
ax[2].set_title('tanh(hyperboilc tangent)')     # 함수 이름 설정
ax[2].set_yticks(np.arange(-1, 1.5, 0.5))     # y축의 눈금 위치를 -1, -0.5, 0, 0.5, 1로 설정
ax[2].grid(True)  # 격자 표시를 활성화

# 네 번째 그래프 - ReLU and softplus 함수
ax[3].plot(x, y_soft, color='red', label='softplus')     # 선의 색상을 빨간색으로 변경, 이름 설정
ax[3].plot(x, y_relu, color='blue', label='ReLU')     # 선의 색상을 파란색으로 변경, 이름 설정
ax[3].set_title('ReLU and softplus')     # 함수 이름 설정
ax[3].legend()  # 범례 추가
ax[3].grid(True)  # 격자 표시를 활성화

plt.show()


