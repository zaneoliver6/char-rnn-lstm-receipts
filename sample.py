from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=100,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)


def sample(args):
    with open(os.path.join('save', 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join('save', 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

        chars=[' ', '\n', '0', '1', '2', 'e', 'i', 'a', '5', '3', 'r', '6', 'n', '4', 'o', '7', 'c', 's', 'b',
               't', '-', 'd', 'g', '8', 'l', 'm', 'u', 'C', 'B', '/', 'p', '.', 'A', 'L', 'R', 'G', '9', 'f',
               'E', 'M', 'y', 'F', 'w', 'Y', 'I', 'k', 'N', 'h', 'z', 'v', 'O', 'U', 'W', 'H', 'T', 'D']
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('save')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            f = open('output.txt','wb')
            f.write(model.sample(sess, chars, vocab, 100, u' ',1).encode('utf-8'))
            f.close()

if __name__ == '__main__':
    main()
