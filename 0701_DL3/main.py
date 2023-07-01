'''
    * 피마 인디언을 대상으로 당뇨병 여부를 측정한 데이터셋
        1) 피마 인디언 데이터
            - 샘플 수 : 768개
            - 속성 (8개)
                - pregnant (과거 임신 횟수)
                - plasma (공복 혈당 농도)
                - pressure (혈압)
                - thickness (심두근 피부 주름 두께)
                - insulin (혈청 인슐린)
                - bmi (체질량 지수)
                - pedigree (당뇨병 가족력)
                - age (나이)
            - 독립 변수
                - diabetes (당뇨 1, 당뇨 아님 0)
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 피마 인디언 당뇨병 데이터셋 불러오기
df = pd.read_csv('./data/pima-indians-diabetes3.csv')

# 처음 10줄 보기
print(df.head(10))

# 정상과 당뇨 환자가 각각 몇 명인지 확인 (정상인 500, 당뇨병 환자 268명)
print(df['diabetes'].value_counts())

# 각 정보별 특징 출력
print(df.describe())

# 각 항목 어느정도의 상관 관계 확인
print(df.corr())

# 그래프로 표현
'''
    hist() 함수 안 
        x축 지정(컬럼)
            - diabetes
            - plasma
        bins : x축을 몇개의 막대로 쪼갤지 정함    
        => plasma 수치가 높을수록 당뇨인 경우가 많음 
'''
colormap = plt.cm.gist_heat     # 그래프 색상 구성 정하기
plt.figure(figsize=(12, 12))    # 그래프 크기 정하기
plt.hist(x=[df.plasma[df.diabetes == 0], df.plasma[df.diabetes == 1]],
         bins=30, histtype='barstacked', label=['normal', 'diabetes'])
plt.legend()
plt.show()

# bmi를 기준으로 각각 정상과 당뇨가 어느 정도 비율로 분포 확인하기
'''
    BMI가 높아질 경우 당뇨의 발병률도 함께 증가하는 추세임
'''
plt.hist(x=[df.bmi[df.diabetes == 0], df.bmi[df.diabetes == 1]],
         bins=30, histtype='barstacked', label=['normal', 'diabetes'])
plt.legend()
plt.show()


#sns.heatmap(data=df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white')
#plt.show()


# 독립변수를 X로 지정
X = df.iloc[:, 0:8]

# 종속변수(당뇨병 유무)를 y로 지정
y = df.iloc[:, 8]

# 모델 구조 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid', name='Dense_3'))
model.summary()

#모델을 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

#모델 학습(실행)
# 100번 반복 실행 결과 약 65%의 정확도를 보임
model.fit(X, y, epochs=100, batch_size=5)
















