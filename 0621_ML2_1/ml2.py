import tensorflow as tf

# 기본적인 모양(shape), 자료형(data type) 출력
data = [
    [1, 2],
    [3, 4]
]

x = tf.constant(data)

print(x)
print(tf.rank(x))           #축(axis)의 개수 출력 = 차원의 개수 출력

data = tf.constant("String")
print(data)

a = tf.constant([5])
b = tf.constant([7])

c = (a + b).numpy()
print(c)
print(type(c))

result = c * 10
tensor = tf.convert_to_tensor(result)
print(tensor)
print(type(tensor))



x = tf.constant([
    [5, 7],
    [1, 2]
])
x_ones = tf.ones_like(x)      # x와 같은 모양과 자료형을 가짐. 값이 1인 텐서 생성
print(x_ones)

# x와 같은 모양을 가지되, 자료형은 float으로 덮어쓰고, 값은 랜덤으로 채우시오
x_rand = tf.random.uniform(shape=x.shape, dtype=tf.float32)
print(x_rand)