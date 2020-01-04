import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import numpy as np
import sys
import os

pathname = os.path.dirname(sys.argv[0])
full_path = os.path.abspath(pathname)

FOLDER_PATH = full_path

with open('embeddings.pickle', 'rb') as handle:
    embeddings = pickle.load(handle)

embeddings_cluster = np.array(list(embeddings.values()))
VOCAB_SIZE = len(embeddings.keys())
EMBEDDING_DIM = embeddings_cluster[0].shape[0]
w2v = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
# region 1 Method 1
#
# X_init = tf.placeholder(tf.float32, shape=(VOCAB_SIZE, EMBEDDING_DIM), name="embedding")
# X = tf.Variable(X_init)
# # Initializer
# init = tf.global_variables_initializer()
# # Start Tensorflow Session
# sess = tf.Session()
# sess.run(init, feed_dict={X_init: w2v})
# # Instance of Saver, save the graph.
# saver = tf.train.Saver()
# writer = tf.summary.FileWriter(TENSORBOARD_FILES_PATH, sess.graph)
#
# # Configure a Tensorflow Projector
# config = projector.ProjectorConfig()
# embed = config.embeddings.add()
# embed.metadata_path = tsv_file_path
# # Write a projector_config
# projector.visualize_embeddings(writer, config)
# # save a checkpoint
# saver.save(sess, TENSORBOARD_FILES_PATH + '/model.ckpt', global_step=VOCAB_SIZE)
# # close the session
# sess.close()
# endregion
# region 2 Method 2

tsv_file_path = FOLDER_PATH + r"\metadata_emb.tsv"
with open(tsv_file_path, 'w+', encoding='utf-8') as file_metadata:
    for i, node in enumerate(embeddings.keys()):
        w2v[i] = embeddings[node]
        embedding = ""
        for s in w2v[i]:
            embedding = embedding + "\t" + str(s)
        file_metadata.write(str(embedding) + '\n')
#endregion