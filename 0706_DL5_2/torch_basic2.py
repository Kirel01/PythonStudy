import torch

a = torch.Tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])
print(a)
print(a.mean())             # 전체 원소에 대한 평균
print(a.mean(dim=0))        # 각 열에 대한 평균 계산
print(a.mean(dim=1))        # 각 행에 대한 평균 계산

print("-----------------")
print(a)
print(a.sum())             # 전체 원소에 대한 합계
print(a.sum(dim=0))        # 각 열에 대한 합계 계산
print(a.sum(dim=1))        # 각 행에 대한 합계 계산

print("-----------------")
print(a)
print(a.max())             # 전체 원소에 대한 최대값
print(a.max(dim=0))        # 각 열에 대한 최대값 계산
print(a.max(dim=1))        # 각 행에 대한 최대값 계산

print("-----------------")
print(a)
print(a.argmax())             # 전체 원소에 대한 최대값의 인덱스
print(a.argmax(dim=0))        # 각 열에 대한 최대값의 인덱스
print(a.argmax(dim=1))        # 각 행에 대한 최대값의 인덱스

print("-----------------")
print(a)
print(a.shape)

# 첫번째 축에 차원 추가
a = a.unsqueeze(0)
print(a)
print(a.shape)

# 네번째 축에 차원 추가
a = a.unsqueeze(3)
print(a)
print(a.shape)

# 크기가 1인 차원 제거
a = a.squeeze()
print(a)
print(a.shape)