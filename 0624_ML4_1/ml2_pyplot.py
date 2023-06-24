'''
*matplotlib.pyplot
    - 데이터 분석의 시각화 방법에서 가장 많이 사용하는 라이브러리
        - 파이썬 오픈소스 라이브러리 중 가장 널리 사용되는 시각화 라이브러리
    - saeborn 같은 다른 양한 시각화 라이브러리들은 matplotlib의 영향을 많이 받음
    - 2002년 부터 만들어짐. MATLAB의 기능들을 파이썬으로 가져옴

*seaborn : statistical data visualization based on matplotlib
    - 2012년 만들어짐 matplotlib를 더 편하게 사용 할 수 있도록 만든 라이브러리
    - numpy, pandas 같은 파이썬 라이브러리를 편하게 시각화하는 것을 중점으로 한 라이브러리
    - DataFrame을 직접적으로 지원
    - EDA 단계에서 빠르게 통계 분석하기에 편리함
'''

import matplotlib.pyplot as plt
import numpy as np

plt.figure()
#plt.plot([1,2,4,8,16])      #라인을 그리는 것이 기본
plt.plot([1,2,3,4,5],[1,2,4,8,16])
#plt.show()

x = ["Americano", "CafeLatte", "CafeMocha", "Macchiato", "Affogato"]
y = [1463, 301, 866, 905, 274]
plt.figure(figsize=(8,6))
plt.bar(x,y, color="pink", label="Sales")
plt.title("Coffe Sales in June", fontsize=20, loc="left" )
plt.xlabel("Menu", fontsize=8)
plt.ylabel("Sales", fontsize=8)
plt.plot(sorted([100,200,400,800,1600], reverse=True),label="Average")
plt.legend()
#plt.grid(alpha=0.3, color='pink', linewidth=0.5)
#plt.ylim(0,2000)
plt.yticks([num for num in range(0,2000,500)])      #y축 숫자 범위
#plt.show()

#scatterplot()

x = np.random.random(1000)
y = np.random.random(1000)

plt.figure(figsize=(6,6))
plt.scatter(x,y, color="orange", marker="^", alpha=0.3)
#plt.show()

'''
1. subplot() 이용해서 여러 개의 plot을 하번에 그림
2. subplot()이용해서 여러 개의 plot을 하나의 figure에 출력할 수 있다
3. plt.subplot(nrows, ncols, index)
               로우 수,컬럼 수,인덱스
'''
plt.figure()
# plt.subplot(2,1,1)
# plt.subplot(2,2,2)

# plt.subplot(2,2,1)
# plt.subplot(2,2,2)
# plt.subplot(2,2,3)
# plt.subplot(2,2,4)

# plt.subplot(2,1,1)
# plt.subplot(2,2,1)
# plt.subplot(2,2,2)


plt.subplot(1,2,1)
plt.plot([0,1, 0.3,0.5])        #x값[0,1,2]에 대응하는 y선을 그림

plt.subplot(3,2,2)
plt.bar([1,2,3],[3,2,1])        #x값[1,2,3에 대응하는 y값[3,2,1]을 막대그래프에

plt.subplot(3,2,4)
plt.plot([4,1,-3])

plt.subplot(3,2,6)

plt.show()

