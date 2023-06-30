from tensorflow import keras
import data_reader

EPOCHS = 20     # 기본값 20

dr = data_reader.DataReader()

'''
    * 활성화 함수
        - sigmoid
        - relu
        - softmax
        - tanh
'''
model = keras.Sequential([
    keras.layers.Dense(3),                      # 첫번째 층은 3개의 유닛을 가지는 레이어 구성
    keras.layers.Dense(128, activation="relu"),  # 두번째 층은 128개의 유닛을 가지는 레이어 구성
    keras.layers.Dense(3, activation="softmax")  # 세번째 층은 3개의 유닛을 가지는 레이어 구성
])

'''
    * 옵티마이저 : 학습을 수행하는 최적화 알고리즘. 
                 경사하강법(Gradient Descent)  
        - Adam 
    * metrics (측정 항목)
        - 어떤 것을 기준으로 성능을 체크할것인지 정의함
        - accuracy : 정확도를 기준으로 모델의 성능을 평가함 
                     예측값이 정답 레이블과 같은 횟수를 계산함 
        - categorical_accuracy : 범주형 정확도              
        
    * 손실함수 : 신경망의 출력과 정답 레이블 간의 차이를 측정하는 함수 
        * sparse_categorical_crossentropy
            -  모델이 어떤 문제를 어떤 카테고리로 잘 분류 했는지 쉽게 확인할수 있음.         
'''
# compile () : 훈련을 위해서 모델을 구성하는 메소드
model.compile(optimizer="adam", metrics=["accuracy"],
              loss="sparse_categorical_crossentropy")


print("**************** Training Strat ========================= ")
# fit() : 훈련 메소드
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
          validation_data=(dr.test_X, dr.test_Y))

# 학습 결과를 그래프로 출력
data_reader.draw_graph(history)