import tensorflow as tf

# 기본적인 모양(shape), 자료형(data type) 출력
data = [
    [1, 2],
    [3, 4]
]

x = tf.constant(data)

print(x)
print(tf.rank(x))           # 차원 출력 