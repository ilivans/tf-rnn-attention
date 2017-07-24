#!/usr/bin/python
"""
Toy example of attention layer use

Train RNN (GRU) on IMDB dataset (binary classification)
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from keras.datasets import imdb

from attention import attention
from utils import *

NUM_WORDS = 10000
INDEX_FROM = 3
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
HIDDEN_SIZE = 150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 256
NUM_EPOCHS = 3  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5

# Load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

# Sequences preprocessing
vocabulary_size = get_vocabulary_size(X_train)
X_test = fit_in_vocabulary(X_test, vocabulary_size)
X_train = zero_pad(X_train, SEQUENCE_LENGTH)
X_test = zero_pad(X_test, SEQUENCE_LENGTH)

# Different placeholders
batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH])
target_ph = tf.placeholder(tf.float32, [None])
seq_len_ph = tf.placeholder(tf.int32, [None])
keep_prob_ph = tf.placeholder(tf.float32)

# Embedding layer
embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

# (Bi-)RNN layer(-s)
rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                        inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
# rnn_outputs, _ = rnn(GRUCell(hidden_size), inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)

# Attention layer
attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)

# Dropout
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# Fully connected layer
W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value, 1], stddev=0.1))
b = tf.Variable(tf.constant(0., shape=[1]))
y_hat = tf.nn.xw_plus_b(drop, W, b)
y_hat = tf.squeeze(y_hat)

# Cross-entropy loss and optimizer initialization
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# Accuracy metric
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))

# Actual lengths of sequences
seq_len_test = np.array([list(x).index(0) + 1 for x in X_test])
seq_len_train = np.array([list(x).index(0) + 1 for x in X_train])

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] / BATCH_SIZE
            for b in range(num_batches):
                x_batch, y_batch = train_batch_generator.next()
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_tr, acc, _ = sess.run([loss, accuracy, optimizer], feed_dict={batch_ph: x_batch, target_ph: y_batch,
                                                                                   seq_len_ph: seq_len, keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] / BATCH_SIZE
            for b in range(num_batches):
                x_batch, y_batch = test_batch_generator.next()
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_test_batch, acc = sess.run([loss, accuracy], feed_dict={batch_ph: x_batch, target_ph: y_batch,
                                                                             seq_len_ph: seq_len, keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        saver.save(sess, "model")
        # saver.save(sess, 'model')
