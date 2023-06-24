'''
    Seaborn
        - 화려한 시각화 기법을 제공
            - histplot, barplot, lineplot...
        - pandas DataFrome과 매우 호환이 잘 됨
'''
import seaborn as sns
import matplotlib.pyplot as plt

# "penguins" 데이터 셋 : 펭귄 종류와 관련된 정보를 포함함
'''
7 columns
    - species : 펭귄의 종류를 나타내는 범주형 변수 (Adelie, Gentoo, Chinstrap)
    - island : 펭귄이 서식하는 섬의 위치를 나타내는 범주형 변수 (Torgersen, Biscoe, Dream)
    - bill_length_mm : 펭귄의 부리 길이를 밀리미터(mm) 단위로 나타내는 수치형 변수
    - bill_depth_mm : 펭귄의 부리 깊이를 밀리미터(mm) 단위로 나타내는 수치형 변수
    - flipper_length_mm : 펭귄의 지느러미 길이를 밀리미터(mm) 단위로 나타내는 수치형 변수
    - body_mass_g : 펭귄의 체중을 그램(g) 단위로 나타내는 수치형 변수
    - sex : 펭귄의 성별을 나나내는 볌주형 변수 (Male, Female)
    
    => 펭귄의 특성과 분포를 다양한 방면에서 분석하는데 사용됨
'''
data = sns.load_dataset("penguins")
data = data.dropna()
print(data)

'''
    1. histplot
        - 가장 기본적으로 사용되는 히스토그램을 출력하는 plot
        - 전체 데이터를 특정 구간별 정보를 확인할 때 사용함
'''

plt.figure(figsize=(6,6))
plt.title("Distribution over body_mass", fontsize=14)
# bins = 막대그래프 개수, hue="species" 기준으로 색깔 부여함, multiple="stack" 따로따로 쌓이게 그림(겹치치 않음)
sns.histplot(data=data, x="body_mass_g", bins=15, hue="species", multiple="stack")
plt.show()

'''
    2. displot
        - distribution들을 여러 subplot들로 나눠서 출력해주는 plot
        - kind를 변경하는 것으로 histplot, kdeplot, ecdfplot 출력이 가능
            - histplot : 히스토그램을 그리는 함수
            - kdeplot : kernael density estimation
                        데이터의 분포를 부드러운 곡선으로 그리는 함수
                        확률밀도함수를 계산하고 이를 곡선으로 시각화함
            - ecdfplot : 누적 분포 함수를 그리는 함수
'''

plt.figure()
# kind="kde" : 밀도 함수, col="island" : 양 옆으로 출력, row="island" : 세로로 출력
sns.displot(data=data, kind="kde", x="body_mass_g", hue="species", col="island") # , row="island")
plt.show()

'''
    3. barplot
        - 어떤 데이터에 대한 값의 크기를 막대로 보여주는 plot (막대 그래프)
        - 가로 / 세로 두 가지로 출력 가능
        - 히스토그램과는 다름        
'''

plt.figure()
# sns.barplot(data=data, x="species", y="body_mass_g")
sns.barplot(data=data, x="body_mass_g", y="species", hue="sex")
plt.show()

'''
    4. countplot
        - 범주형 속성을 가지는 데이터들의 histogram을 보여주는 plot
        - 종류별 count를 보여주는 방법
        - set_palette() : 색상 팔레트 설정 함수
            - Set1 : 4가지의 서로 다른 색상을 가진 팔렛트
            - Set2 : 8가지의 서로 다른 색상을 가진 팔렛트
            - Set3 : 12가지의 서로 다른 색상을 가진 팔렛트
            - viridis : 연속적인 색상 그라데이션을 가진 팔렛트
'''

plt.figure(figsize=(8,6))
sns.set_palette("Set1")
sns.countplot(data=data, x="sex", hue="species")
plt.show()

'''
    5. boxplot
        - 데이터의 각 종류별로 사분위 수를 표시하는 plot
        - 특정 데이터의 전체적인 분포를 확인하기 좋은 시각화 기법
        - box와 전체 range의 그림을 통해 outlier를 잦기 쉬움
'''

plt.figure(figsize=(8,6))
sns.set_palette("Set2")
sns.boxplot(data=data, x="bill_depth_mm", y = "species", hue="sex")
plt.show()

'''
    6. violinplot
        - 데이터에 대한 분포 자체를 보여주는 plot
        - boxplot과 비슷하지만, 전체 분포에 대한 그림을 보여줌
'''

plt.figure(figsize=(8,6))
sns.set_palette("Set3")
sns.violinplot(data=data, x="bill_depth_mm", y = "species", hue="sex")
plt.show()

'''
    7. lineplot
        - 특정 데이터를 x,y로 표시하여 관계를 확인할 수 있음
        - 선 그래프
        - 수치형 자료들 간의 경향을 파악할 때 많이 사용
'''

plt.figure(figsize=(8,6))
sns.set_palette("Set1")
sns.lineplot(data=data, x="body_mass_g", y = "bill_depth_mm")
plt.show()

'''
    8. pointplot
        - 특정 수치 데이터를 error bar와 함께 출력해주는 plot
        - 데이터와 error bar를 한번에 찍어주기 떄문에, 특정 지표들만 사용하는 것이 좋음
'''

plt.figure(figsize=(8,6))
sns.pointplot(data=data, x="island", y = "bill_length_mm")
plt.show()

'''
    9. scatterplot
        - lineploto과 비슷하게 x,y에 대한 전체적인 분포를 확인하는 plot
        - lineplot과 경향성에 초점을 둔다면, scatterplot은 데이터 그 자체가 퍼져있는 모양에 중점을 둠
'''

plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x="flipper_length_mm", y = "body_mass_g", hue="sex")
plt.show()

'''
    10. pairplot
        - 주어진 데이터의 각 feature들 사이의 관계를 표시하는 plot
        - scatterplot, FaceGrid, kdeplot을 이용하여 feature들 간의 관계를 잘 보여줌
        - 각 feature에 대해 계산된 모든 결과를 보여줌
            - feature가 많은 경우 사용하기 적합하지 않음
'''

plt.figure(figsize=(10,6))
sns.pairplot(data=data, hue="sex")
plt.show()

'''
    11. heatmap
        - 정사각형 그림에 데이터에 대한 정도 차이를 색 차이로 보여주는 plot
        - 열 화상 카메라로 사물을 찍은 것처럼 정보의 차이를 보여줌
        - pairplot과 비슷하게 feature간의 관계를 시각화할 때 많이 사용함
        - 상관관계
            - x값의 변화에 따라 y값의 선형적으로 변화하는지를 측정한 지표
            - ex) 아이스크림 판매량 증가 <--> 상어에 물린 사람 수
'''
# 각 feature 간 상관관계를 파악하기 위해 correlation matrix 생성
corr = data.corr()
print(corr)

plt.figure(figsize=(8,8))
sns.heatmap(data=corr)
plt.show()