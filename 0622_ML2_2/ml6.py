import tensorflow as tf

# 같은 크기를 가진 두 개의 텐서에 대해 사칙 연산 가능
# 기본적으로 요소별(element-wise) 연산

a = tf.constant ([
    [1,2],
    [3,4]
])

b = tf.constant ([
    [5,6],
    [7,8]
])

print(a+b)
print(a-b)
print(a*b)
print(a/b)


# 행렬 곱(matrix multiplication) 수행
print(tf.matmul(a,b))
