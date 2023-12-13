import tensorflow as tf

# Check for TensorFlow GPU access
print(tf.config.list_physical_devices())

# See TensorFlow version
print(tf.__version__)

tf.config.list_physical_devices('GPU')

with tf.device("/GPU"):
    a = tf.random.normal(shape=(2,), dtype=tf.float32)
    b = tf.nn.relu(a)

print(a)
print(b)
