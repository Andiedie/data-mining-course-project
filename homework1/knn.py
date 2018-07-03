import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--output_path', default='./knn_output.csv',
                    type=str, help='path to save output csv')
parser.add_argument('--k', default=1,
                    type=int, help='k nearest neighbors')


def main(argv):
    args = parser.parse_args(argv[1:])
    output_path = os.path.abspath(args.output_path)
    data_dir = os.path.abspath('./dataset')
    (train_X, train_Y), test_X = load_data(data_dir)

    dataset = tf.data.Dataset.from_tensor_slices(test_X)
    dataset = dataset.batch(args.batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    num_features = train_X.shape[1]
    train_x = tf.placeholder(shape=[None, num_features], dtype=tf.float64)
    # test_x = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
    distance = tf.reduce_sum(
        tf.abs(tf.subtract(train_x, tf.expand_dims(next_batch, 1))), axis=2)
    _, indices = tf.nn.top_k(tf.negative(distance), args.k)

    prediction = tf.reduce_mean(tf.gather(train_Y, indices), axis=1)

    with tf.Session() as sess:
        result = []
        number_test = test_X.shape[0]
        try:
            while True:
                preds = sess.run(prediction, feed_dict={
                    train_x: train_X
                })
                result.extend(preds)
                print('%d / %d' % (len(result), number_test))
        except tf.errors.OutOfRangeError:
            print('Done')

        pd.Series(result).to_csv(
            path=output_path,
            header=['reference'],
            index_label='id'
        )


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    train = pd.read_csv(open(train_path))
    train.pop('id')
    train_X, train_Y = train, train.pop('reference')

    test_X = pd.read_csv(open(test_path))
    test_X.pop('id')

    return (train_X.values, train_Y.values), test_X.values


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
