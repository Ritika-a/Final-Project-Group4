import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
# import urllib.request as urllib2
import urllib
import time
import random
whole_start = time.time()

random.seed(5)

tf.reset_default_graph()

url_response = urllib.urlretrieve("https://storage.googleapis.com/ml2-group4-project/all_images.npy", "all_images.npy")
x = np.load("all_images.npy")

url_response = urllib.urlretrieve("https://storage.googleapis.com/ml2-group4-project/all_labels.npy", "all_labels.npy")
y = np.load("all_labels.npy")


y = y.astype(np.float32)

print(x.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)


X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 5])
hold_prob = tf.placeholder(tf.float32)


def conv_network(xdata):

    w1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1), name="weight1")
    b1 = tf.Variable(tf.constant(0.1, shape=[32]), name="bias1")

    w1_summary = tf.summary.histogram('w1_histogram_summary', w1)
    b1_summary = tf.summary.histogram("b1_histogram_summary", b1)

    conv1 = tf.nn.conv2d(xdata, w1, strides=[1, 1, 1, 1], name="convolution1", padding='SAME')
    conv_layer1 = tf.nn.relu(conv1 + b1)

    pool1 = tf.nn.max_pool(conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max1')
#------------------------------------------LAYER2 -----------------------------------------------
    w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="weight2")
    b2 = tf.Variable(tf.constant(0.1, shape=[64]), name="bias2")

    w2_summary = tf.summary.histogram('w2_histogram_summary', w2)
    b2_summary = tf.summary.histogram("b2_histogram_summary", b2)

    conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], name="convolution2", padding='SAME')
    conv_layer2 = tf.nn.relu(conv2 + b2)

    pool2 = tf.nn.max_pool(conv_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max2')

# ---------------------------------------LAYER 3-----------------------------------------------------

    w3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1), name="weight3")
    b3 = tf.Variable(tf.constant(0.1, shape=[128]), name="bias3")

    # w2_summary = tf.summary.histogram('w2_histogram_summary', w2)
    # b2_summary = tf.summary.histogram("b2_histogram_summary", b2)

    conv3 = tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], name="convolution3", padding='SAME')
    conv_layer3 = tf.nn.relu(conv3 + b3)

    pool3 = tf.nn.max_pool(conv_layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max3')
#--------------------------------------------FLATTENING AND FULLY CONNECTED-------------------------------------------------------------------

    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 128])

    input_size = int(pool3_flat.get_shape()[1])
    w4 = tf.Variable(tf.truncated_normal([input_size, 2000]), name='weight4')
    b4 = tf.Variable(tf.constant(0.1, shape=[2000]), name="bias4")

    vector_1 = tf.matmul(pool3_flat, w4) + b4
    fully_connected_1 = tf.nn.relu(vector_1)

# ------------------------------------------------------------------------
    input_flat = int(fully_connected_1.get_shape()[1])

    w5 = tf.Variable(tf.truncated_normal([input_flat, 5]), name='weight5')
    b5 = tf.Variable(tf.constant(0.1, shape=[5]), name="bias5")

    dropout_layer1 = tf.nn.dropout(fully_connected_1, keep_prob=hold_prob)
    y_pred = tf.matmul(dropout_layer1, w5) + b5

    return y_pred
# ------------------------------------------------------------------------------------------------


logits = conv_network(X)


loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)

cross_entropy = tf.reduce_mean(loss)

# ------------------------------------------------------------------------
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train_op = optimizer.minimize(cross_entropy)

match = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(match, tf.float32))

loss_summary = tf.summary.scalar(name="loss_summary", tensor=cross_entropy)

accuracy_summary = tf.summary.scalar(name='accuracy_summary', tensor=accuracy)

#HYPERPARAMETERS

batch_size = 100
test_batch_size = 100
test_data_length = len(X_test)
data_length = len(X_train)

amount_batches = int(data_length/batch_size)
amount_test_batches = int(test_data_length/test_batch_size)

print(data_length)
print(amount_batches)


epoch_size = 100

merged_summary = tf.summary.merge_all()
summary_dir = './summary_dir'

begin = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(summary_dir+'/train_graphs', graph=sess.graph)
    test_writer = tf.summary.FileWriter(summary_dir+'/test_graphs')

    for epoch in range(epoch_size):
        idx = 0

        for batch in range(amount_batches):
            images_batch = X_train[idx:idx+batch_size]
            labels_batch = y_train[idx:idx+batch_size]
            idx = idx + batch_size

            sess.run(train_op, feed_dict={X: images_batch, y_true: labels_batch, hold_prob: 0.7})
            loss_train = sess.run(cross_entropy, feed_dict={X: images_batch, y_true: labels_batch, hold_prob: 0.7})

        i = 0
        accuracy_per_batch = []
        for test_batch in range(amount_test_batches):
            images_test_batch = X_test[i:i+test_batch_size]
            labels_test_batch = y_test[i:i+test_batch_size]

            acc = sess.run(accuracy, feed_dict={X: images_test_batch, y_true: labels_test_batch, hold_prob: 1})
            accuracy_per_batch.append(acc)

        mean_accuracy = np.mean(accuracy_per_batch)
        print("test_accuracy " + str(mean_accuracy))
        print("Loss :" + str(loss_train))
        end = time.time()-begin
        print("Epoch number: " + str(epoch) + " Epoch time: " + str(round(end, 2)))
        print("\n")

        summary = sess.run(merged_summary, feed_dict={X: images_batch, y_true: labels_batch, hold_prob: 0.7})
        train_writer.add_summary(summary, global_step=epoch)
        summary1 = sess.run(merged_summary, feed_dict={X: images_test_batch, y_true: labels_test_batch, hold_prob: 1})
        test_writer.add_summary(summary1, global_step=epoch)

whole_end = time.time()-whole_start
print("Total time taken is: " + str(whole_end))

# Run tensorboard --logdir=./summary_dir in the same folder as .py file.
# Then open chromium browser and copy the link show after running the command above





