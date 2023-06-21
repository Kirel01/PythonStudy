import tensorflow as tf

tensor = tf.constant([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print(tensor[0])        # first row
print(tensor[:, 0])     # first column
print(tensor[:, -1])    # last column
