"""
Toy example of attention layer use.

Train RNN (GRU) on IMDB dataset (binary classification).
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn_cell import GRUCell
from keras.datasets import imdb

from attention import attention
from utils import *


# Load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()

# Sequences preprocessing
vocabulary_size = get_vocabulary_size(X_train)
X_test = fit_in_vocabulary(X_test, vocabulary_size)
sequence_length = 250
X_train = zero_pad(X_train, sequence_length)
X_test = zero_pad(X_test, sequence_length)

# Different placeholders
batch_ph = tf.placeholder(tf.int32, [None, sequence_length])
target_ph = tf.placeholder(tf.float32, [None])
seq_len_ph = tf.placeholder(tf.int32, [None])
keep_prob_ph = tf.placeholder(tf.float32)

# Embedding layer
embed_dim = 100
embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, embed_dim], -1.0, 1.0), trainable=True)
batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

# (Bi-)RNN layer(-s)
hidden_size = 150
# birnn_outputs, _ = bi_rnn(GRUCell(hidden_size), GRUCell(hidden_size),
#                           inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
rnn_outputs, _ = rnn(GRUCell(hidden_size), inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)

# Attention layer
attention_size = 50
attention_output = attention(rnn_outputs, attention_size)

# Dropout
keep_prob = 0.5
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# Fully connected layer
W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value, 1], stddev=0.1))
b = tf.Variable(tf.constant(0., shape=[1]))
y_hat = tf.nn.xw_plus_b(drop, W, b)
y_hat = tf.squeeze(y_hat)

# Cross-entropy loss and optimizer initialization
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_hat, target_ph))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# Accuracy metric
accuracy = 1. - tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), target_ph), tf.float32))

# Actual lengths of sequences
seq_len_test = np.array([list(x).index(0) + 1 for x in X_test])
seq_len_train = np.array([list(x).index(0) + 1 for x in X_train])

# Train batch generator
batch_size = 256
train_batch_generator = batch_generator(X_train, y_train, batch_size)
test_batch_generator = batch_generator(X_test, y_test, batch_size)

num_epochs = 10
delta = 0.5
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print("Start learning...")
    for epoch in range(num_epochs):
        loss_train = 0
        loss_test = 0
        accuracy_train = 0
        accuracy_test = 0

        print("epoch: {}".format(epoch))

        # Training
        num_batches = X_train.shape[0] / batch_size
        for b in range(num_batches):
            x_batch, y_batch = train_batch_generator.next()
            seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
            loss_tr, acc, _ = sess.run([loss, accuracy, optimizer], feed_dict={batch_ph: x_batch, target_ph: y_batch,
                                                                               seq_len_ph: seq_len, keep_prob_ph: keep_prob})
            accuracy_train += acc
            loss_train = loss_tr * delta + loss_train * (1 - delta)
        accuracy_train /= num_batches

        # Validating
        num_batches = X_test.shape[0] / batch_size
        for b in range(num_batches):
            x_batch, y_batch = test_batch_generator.next()
            seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
            loss_test_batch, acc = sess.run([loss, accuracy], feed_dict={batch_ph: x_batch, target_ph: y_batch,
                                                                         seq_len_ph: seq_len, keep_prob_ph: 1.0})
            accuracy_test += acc
            loss_test += loss_test_batch
        accuracy_test /= num_batches
        loss_test /= num_batches

        print("\t train\tloss: {:.3f}\t acc: {:.3f}".format(loss_train, accuracy_train))
        print("\t test\tloss: {:.3f}\t acc: {:.3f}".format(loss_test, accuracy_test))
