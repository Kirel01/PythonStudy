import os
import random
import time

# INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import pip

    pip.main(['install', 'matplotlib'])
    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        time.sleep(2)
        from matplotlib import pyplot as plt

try:
    import numpy as np
except ModuleNotFoundError:
    import pip

    pip.main(['install', 'numpy'])
    try:
        import numpy as np
    except ModuleNotFoundError:
        time.sleep(2)
        import numpy as np

class DataReader():
    def __init__(self):
        # 데이터를 저장할 변수들
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_data()

        # 데이터 읽기 완료
        # 읽어온 데이터 정보 출력
        print("\n\n 데이터 읽기 완료")
        print("훈련(Traing) X size : " + str(self.train_X.shape) )
        print("훈련(Traing) Y size : " + str(self.train_Y.shape) )
        print("테스트(Test) X size :  " + str(self.test_X.shape) )
        print("테스트(Test) Y size :  " + str(self.test_Y.shape) )

    # 데이터를 읽어오기 위한 메서드
    def read_data(self):

        fileName = os.listdir("data")[0]
        file = open("data/" + fileName, encoding="utf-8")

        #헤더 제거
        file.readline()

        #데이터와 레이블을 저장하기 위한 비어있는 리스트
        data = []
        #파일을 한줄씩 읽어오기
        for line in file:
            # 컴마를 기준으로 split() 실행
            splt = line.split(",")
            # split 결과물을 정리해 X값과 Y값으로 추려내기
            x, compulsory = self.process_data(splt)
            # 추려낸 데이터를 저장
            data.append((x, compulsory))

        # 학습하기 전 전체 데이터를 섞음
        random.shuffle(data)

        X = []      # 독립변수
        y = []      # 종속변수(정답)

        for el in data:
            X.append(el[0])
            y.append(el[1])

        #넘파이 배열로 생성함
        X = np.asarray(X)
        y = np.asarray(y)

        # 훈련데이터 셋 : 테스트데이터 셋 = 8 : 2
        train_X = X[:int(len(X) * 0.8)]
        train_Y = y[:int(len(y) * 0.8)]

        test_X = X[int(len(X) * 0.8): ]
        test_Y = y[int(len(y) * 0.8): ]

        return train_X, train_Y, test_X, test_Y

    # split()한 값을 정리하기 위한 메서드
    def process_data(self, splt):
        # 읽어온 splt 값에서 학교, 성별, 키, 몸무게만 추출
        school = splt[9]            # 종속변수

        gender = splt[13]                       # 독립변수
        height = float(splt[15]) / 194.2        # 독립변수 (최대값으로 나눔)
        weight = float(splt[16]) / 130.7        # 독립변수 (최대값으로 나눔)

        # 추려낸 데이터를 저장할 비어있는 리스트
        data2 = []
        data2.append(height)
        data2.append(weight)

        # 성별이 남자일 경우 1, 여자일 경우 0
        if gender == "남":
            data2.append(1)
        else:
            data2.append(0)

        # 초등학교, 중학교, 고등학교 정보를 정리
        # compulsory는 레이블(예측값)
        if school.endswith("초등학교"):
            compulsory = 0
        elif school.endswith("중학교"):
            compulsory = 1
        elif school.endswith("고등학교"):
            compulsory = 2

        # 결과물 리턴
        return data2, compulsory


# 학습 이력(history) 정보 쉽게 확인
# 훈련 데이터와 검증 데이터에 대한 손실 함수값의 history 시각화
# 훈련 데이터와 검증 데이터에 대한 정확도 값의 시각화
def draw_graph(result):
    train_result = result.history["loss"]   # history 딕셔너리에서 "loss"를 추출
    validation_result = result.history["val_loss"]  # history 딕셔너리에서 "val_loss"를 추출

    fig = plt.figure(figsize=(8, 8))      # 그래프 크기 설정
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss Function")
    plt.plot(train_result, "red")   # 훈련 데이터에 대한 손실 함수 값의 히스토리 빨간색 표시
    plt.plot(validation_result, "blue")  # 검증 데이터에 대한 손실 함수 값의 히스토리 파란색 표시
    fig.savefig("train_history.png")

    train_result = result.history["accuracy"]
    validation_result = result.history["val_accuracy"]

    fig = plt.figure(figsize=(8, 8))      # 그래프 크기 설정
    plt.title("Accuracy History")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy Function")
    plt.plot(train_result, "red")   # 훈련 데이터에 대한 정확도 값의 히스토리 빨간색 표시
    plt.plot(validation_result, "blue")  # 검증 데이터에 대한 정확도 값의 히스토리 파란색 표시
    fig.savefig("accuracy_history.png")







