import tensorflow as tf

a = tf.constant([2])
b = tf.constant([5.0])

print(a.dtype)
print(b.dtype)

#텐서 a를 float32 형식으로 변경한 뒤에 더하기 수행
print(tf.cast(a, tf.float32) + b)

a = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
b = tf.reshape(a, (4, 2))
print(b)

# a와 b는 서로 다른 객체
a.assign_add([1, 1, 1, 1, 1, 1, 1, 1])
print(a)
print(b)

print("=====================================")
a = tf.random.uniform((64, 32, 3))
print(a)

print("=====================================")
b = tf.transpose(a, perm=[2, 1, 0])  # 차원 자체를 교환
                        # 2번째 축, 1번째 축, 0번째 축의 형태가 됨
print(b)