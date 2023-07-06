import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as tr
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

'''
    * torchvision
        - 파이토치에서 제공하는 데이터 셋들이 모여있는 패키지 
            - Fashion-MNIST data 
            
    * DataLoader
        - Dataset을 model에 공급할수 있도록 iterable 객체로 감싸주는 클래스
        
    * 신경망 구축 및 실행순서
        1) Fashion-MNIST data 불러오기
        2) 데이터 시각화
        3) DataLoader 만들기
            - Custom Dataset, Data Loader 만들기 
        4) Model Class 만들기 
        5) Training / Validation(Test) Function                                        
'''

training_data = datasets.FashionMNIST(
    root="data",        # 데이터를 저장할 디렉토리 지정
    train=True,         # 훈련데이터 로드
    download=True,      # 데이터셋을 인터넷에서 다운로드 지정
    transform=ToTensor()    # 이미지를 PyTorch의 텐서로 변환하는 함수
)

test_data = datasets.FashionMNIST(
    root="data",        # 데이터를 저장할 디렉토리 지정
    train=False,         # 훈련데이터 로드
    download=True,      # 데이터셋을 인터넷에서 다운로드 지정
    transform=ToTensor()    # 이미지를 PyTorch의 텐서로 변환하는 함수
)

print(training_data)
print(type(training_data))

print("-" * 50)
print(training_data[0])

#데이터 시각화
labels_map = {
    0: 	"T-shirt",
    1:	"Trouser",
    2:	"Pullover",
    3:	"Dress",
    4:	"Coat",
    5:	"Sandal",
    6:	"Shirt",
    7:	"Sneaker",
    8:	"Bag",
    9:	"Ankle boot"
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3,3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")  # 텐서의 크기에서 차원의 크기가 1인것 제거.

plt.show()