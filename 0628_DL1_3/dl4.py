import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dl1_perceptron import Perceptron


class Perceptron(object):
    '''
        매개변수
            - eto : 학습률
            - n_iter : 훈련 데이터셋 반복 횟수
            - random_seed :
    '''
    def __init__(self, eta=0.01, n_iter=50, random_seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_seed = random_seed

    '''
        훈련 데이터 학습함수의 매개변수
            - X : 훈련 데이터 [n_sample, n_features]
                - n_sample개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
            - y : target 값 [n_sample]
    '''
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= 1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update + xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


    def net_input(self, X):
        '''입력 계산'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        '''클래스 레이블 반환(예측값 출력)'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    ptron = Perceptron(eta=0.1, n_iter=1.0)

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

