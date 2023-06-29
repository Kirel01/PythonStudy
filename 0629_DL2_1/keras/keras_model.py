import numpy as np
import tensorflow as tf

#Sequential 모델 생성
model = tf.keras.models.Sequential()

#Sequential 모델에 add()함수를 이용하여 레이어(은닉층) 추가
model.add(tf.keras.layers.Dense(units=2, input_shape=(2,),
                                activation='sigmoid'))
#Sequential 모델에 add()함수를 이용하여 레이어(출력층) 추가
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compile() 함수 호출하여 Sequential 모델을 컴파일
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.3))
'''
    * 하이퍼 매개변수(파라미터)
        - 신경망의 학습률이나 가중치, 은닉층의 개수, 유닛의 개수, 미니 배치의 크기 등
        - 기본값 사용 혹은 사용자가 하이퍼 매개변수를 지정 
'''
model.summary()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 학습 수행
model.fit(X, y, batch_size=1, epochs=10000)

# 모델 테스트 (예측)
'''
첫번째 입력 데이터 [0, 0]에 대한 모델의 예측값. 0을 예측함
'''
print(model.predict(X))
