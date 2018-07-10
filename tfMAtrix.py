import tensorflow as tf

### Step1: build the structure
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1, matrix2)


# Controled by session to display different specific elements in the structure

### method1
sess = tf.Session()
results = sess.run(product)
print(results)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)