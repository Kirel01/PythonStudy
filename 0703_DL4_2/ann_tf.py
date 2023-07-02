# 텐서플로 이용한 인공신경망 구현

import tensorflow as tf

# MNIST data set 로드하기
'''
    MNIST 데이터 세트
        - load_data() 함수 이용하여 데이터 가져옴
        - 손으로 쓴 글씨(0~9까지)의 이미지로 구성됨
        - 훈련 데이터와 테스트 데이터를 반환 
        - 각각 입력 이미지와 해당 이미지의 레이블로 구성됨
'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 이미지 데이터를 float32로 변경하기
'''
    astype()함수 사용하여 이미지 데이터 자료형 변경
    => 신경망 모델의 입력으로 사용하기 위함 
'''
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

# 28 * 28 = 784  펼치기(reshape)
'''
    2차원 배열에서 1차원 배열로 변환 (전처리)
        - MNIST 데이터셋 이미지 : 각각 28 * 28 크기의 2차원 배열임 
        - 신경망 모델 : 입력으로 사용하기 위해서는 1차원 배열 형태로 변환해야 함 
        - -1 : 행의 크기는 자동으로 결정,
          784 : 열의 크기로 지정    
        - x_train이 60000개 이미지이고, 각 이미지 크기가 28*28이라면
          reshape([-1, 784])를 적용하면 60000 * 784 크기의 2차원 배열로 변환됨
          - 각 행은 하나의 이미지에 해당됨
          - 픽셀 값들이 일련로 나열된 형태로 저장됨    
'''
x_train,x_test =  x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

# 입력값의 정규화(normalize) : [0, 255] => [0, 1]
'''
    * MNIST 데이터 세트의 이미지는 픽셀 값(0~255)
        - 이미지 데이터를 신경망 모델에 입력으로 사용할 때, 
          일반적으로 픽셀값 범위를 0부터 1사이로 제한하는 것이 좋음 
          => 가중치 업데이트와 같은 연산에서도 안정성을 높여줌 
'''
x_train,x_test =  x_train / 255, x_test / 255

# One-hot encoding 레이블로 변경
'''
    one_hot은 다중 클래스 분류 문제를 해결하기 위함
        - 신경망 모델에서 다중 클래스 분류를 수행하기 위해서는 출력값 적절하게 설정해야함 
        - 각 클래스에 대한 확률을 출력하기 위해 범주형 데이터 형식인 one-hot 인코딩을 사용함
        
        - 각 클래스를 해당하는 인덱스에 1을 표시하고, 나머지 인덱스는 0을 표시하는 방식
            - 각 클래스의 레이블을 벡터 형태로 표시함 
            - 변환된 레이블 벡터는 신경망 모델의 출력과 비교하여 
              정확성을 계선하거나 손실함수를 계산하는 작업에 사용됨 
'''
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

#print(y_train)
#print(x_train)

#-----------------------------------------
#학습에 사용되는 파라미터 정의
'''
    * 학습률 
        - 경사 하강법(gradient descent) 알고리즘에서 한번에 업데이트되는 
          가중치의 크기를 조절하는 역할을 함 
        - 학습률은 0 ~ 1 사이의 값으로 설정됨 
            - 너무 적으면 학습 속도가 느려짐, 너무 크면 문제가 됨   
            
    * 학습 횟수
        - 전체 데이터 세트를 몇 번 반복하여 학습할지를 나타냄
        - 너무 큰 값은 과적합(overfitting)의 가능성을 증가시킴    
        
    * 배치 개수
        - 한번에 모든 데이터를 처리하는 것이 아니라 
          일부 데이터를 묶어서 처리함 ==> 메모리 사용량을 줄이고, 계산 속도 향상시킴 
          
    * display_step
        - 학습 과정에서 손실 함수의 값은 주로 모니터링 지표중 하나임 
        - 몇 번째 학습 단계마다 손실 함수의 값을 출력할지를 결정함 
        - 1 : 매 학습 단계마다 출력됨                
        
    * input_size
        - 입력의 크기는 이미지의 특징 개수임 
        - MNIST 데이터 세트는 이미지 크기가 28 * 28 = 784 픽셀임 
        
    * hidden1_size
        - 은닉층(hidden layer)의 크기
        - 신경망 모델의 은닉층은 입력과 출력 사이에 있는 층으로,
          중간에 특성을 학습하고 추출하는 역할 함     
          
    * output_size
        - 모델의 출력층 노드의 크기 
        - 숫자 0부터 9까지 총 10개의 클래스를 분류 (다중 클래스 분류)                                 
'''
learning_rate = 0.001       # 학습률
num_epochs = 30             # 학습 횟수
batch_size = 256            # 배치 개수
display_step = 1            # 손실함수 출력 주기
input_size = 784            # 28 * 28
hidden1_size = 256          # 은닉층의 크기
hidden2_size = 256
output_size = 10            # 출력층의 크기



# 데이터 섞기(shuffle)
'''
    * from_tensor_slices()
        - 텐서들을 슬라이스하여 데이터셋을 생성하는 함수 
        - 이미지와 해당 레이블로 이루어진 훈련 데이터
            - 묶어서 데이터셋을 생성함 
    * shuffle() 
        - 데이터셋을 섞는 역할
        - 훈련 데이터를 무작위로 섞음 
            - 모델이 데이터의 순서에 의존하지 않고 학습되도록 하기 위함 
    * batch() 
        - 데이터셋을 배치로 나누는 역할
        - 한번의 업데이트 단계에서 여러 개의 데이터 샘플을 함께 처리할수 있음 
        - 배치처리를 통해 메모리 효율성과 계산 속도 개선함            
'''
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)

# ANN 클래스 정의
'''
   * 네트웍의 가중치와 편향 변수 초기화
    - tf.Variable()으로 정의
    - 첫번째 은닉층의 가중치와 편향
    
   *  return logits
    - 신경망의 예측값 
    
   * 데코레이터(decorator)
    - 파이썬에서 함수나 클래스를를 수정하거나 기능을 추가할 때 사용되는 문법적인 요소임.
    - 기존의 함수나 클래스를 변경하지 않으면서 새로운 기능을 추가함   
    
'''
class ANN(object):
    def __init__(self):
        self.W1 = tf.Variable(tf.random.normal(shape=[input_size, hidden1_size]))
        self.b1 = tf.Variable(tf.random.normal(shape=[hidden1_size]))
        self.W2 = tf.Variable(tf.random.normal(shape=[hidden1_size, hidden2_size]))
        self.b2 = tf.Variable(tf.random.normal(shape=[hidden2_size]))
        self.W_output = tf.Variable(tf.random.normal(shape=[hidden2_size, output_size]))
        self.b_output = tf.Variable(tf.random.normal(shape=[output_size]))

    def __call__(self, x):
        H1_output = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        H2_output = tf.nn.relu(tf.matmul(H1_output, self.W2) + self.b2)
        logits = tf.matmul(H2_output, self.W_output) + self.b_output

        return logits

# Loss Function(손실함수) Cross-Entropy
'''
    * 데코레이터(decorator) 
        - 파이썬에서 함수나 클래스를 수정하거나 
          기능을 추가할 때 사용되는 문법임 
    * TensorFlow에서의 @tf.function 데코레이터
        - 그래프로 함수를 변환하여 성능을 향상시키는 역할을 함  
        - 반복적인 연산을 줄여서 모델의 실행 속도를 개선함   
        
    * reduce_mean()
        - 평균 손실은 모델의 전체 손실을 나타내는 지표로 사용됨 
        - 텐서의 손실 평균을 계산하는 함수임
            - 교차 엔트로피 손실의 평균을 계산해서 반환함 
            - 예측값 logits와 실제 레이블 y 사이의 교차 엔트로피 손실 계산
               
'''
@tf.function
def cross_entropy(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# Adam
'''
    * Adam
        1) 최적화 알고리즘을 구현한 TensorFlow의 옵티마이저 클래스임.
        2) 자동 미분을 활용하여 모델의 가중치를 업데이트하는데 사용되는 알고리즘임
        3) 학습률 - 각 파라미터 업데이트의 크기를 결정하는 요소
'''
optimizer = tf.optimizers.Adam(learning_rate)

# 최적화를 위한 함수
'''
    train_step() 함수
        1) 모델, 입력 데이터 x, 실제 레이블 y를 사용하여
           한단계의 학습을 수행하고, 모델의 변수들을 업데이터하는 역할을 함  
        2) tf.GradientTape()
            - 블록 안에서의 모델의 연산들을 기록함   
            - 입력 데이터x를 받아서 예측값 y_pred을 계산함 
            - 이것을 손실 함수에 적용하여 손실 loss를 계산함 
        3) tape.gradient() 
            - 손실에 대한 모델의 변수들의 그레디언트(기울기)를 계산함   
        4) optimizer.apply_gradients()
            - 계산된 gradients(기울기)를 사용하여 모델의 변수들을 업데이트함 
            - 옵티마이저에 의해 그래디언트 기반 최적화 알고리즘을 수행.               
'''
@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = cross_entropy(y_pred, y)
    gradients = tape.gradient(loss, vars(model).values())
    optimizer.apply_gradients(zip(gradients, vars(model).values()))

# 모델 정확도 정의
'''
    * 예측값 y_pred와 실제 레이블 y를 사용하여
    모델의 정확도를 계산하는 역할
    => 모델의 성능을 평가하고 모델의 예측 결과를 정량적으로 측정할 수 있음 
        -  tf.equal()
            - tf.argmax() : 가장 큰 원소의 인덱스를 가져옴 
            - 예측값과 실제 레이블이 일치하는지 여부를 계산함 
                - 일치하면 True 반환, 일치하지 않으면 False 반환 
        - tf.cast()
            - 데이터 타입을 floast32 타입으로 변경
            - True를 1.0으로, False를 0.0으로 변환 (정확도 계산하기 위함)                    
'''
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy





# ANN 모델 객체 생성
ANN_model = ANN()

# model optimization
for epoch in range(num_epochs):
    average_loss = 0
    # 총 배치 개수 = 전체 데이터 셋을 배치 크기로 나눔
    total_batch = int(x_train.shape[0] / batch_size)
    # optimize for every batch
    for batch_x, batch_y in train_data:
        # 손실(loss) 업데이트
        _, current_loss = train_step(ANN_model, batch_x, batch_y), \
            cross_entropy(ANN_model(batch_x), batch_y)
        # 평균 손실값 계산
        average_loss += current_loss / total_batch
    # 매 epoch마다 출력
    if epoch % display_step == 0:
        print("Epoch: %d, Loss: %f" %((epoch+1), average_loss))

# 정확도 출력
print("Accuracy : %f" %compute_accuracy(ANN_model(x_test), y_test))




