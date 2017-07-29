import tensorflow as tf
import os
from six.moves import cPickle
from model import Model

sess = tf.InteractiveSession()
#input = tf.placeholder(dtype=tf.int32, shape=[1,1], name='input')

with open(os.path.join('save', 'config.pkl'), 'rb') as f:
    saved_args = cPickle.load(f)

model = Model(saved_args, training=False)

tf.global_variables_initializer().run()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('save')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    graph_def = sess.graph_def
    tf.train.write_graph(graph_def,"save/","mobilernn.pbtxt")
    saver.save(sess,"save/mobilernn.ckpt")
