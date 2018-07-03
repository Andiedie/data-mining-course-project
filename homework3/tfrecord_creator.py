import tensorflow as tf
import os
import dataset_util
import pandas as pd
import pickle

_TRAIN_RATE = 0.9

def make_example(filename, label):
    with tf.gfile.GFile(filename, 'rb') as fid:
        image = fid.read()
    return tf.train.Example(features=tf.train.Features(feature={
        'label': dataset_util.int64_feature(label),
        'image': dataset_util.bytes_feature(image)
    }))


def writeToFile(writer, df):
    for _, row in df.iterrows():
        example = make_example(
            './dataset/train/' + row['Image'], row['Cloth_label'])
        writer.write(example.SerializeToString())


def main(argv):
    print('read csv file')
    csv = pd.read_csv('./dataset/train.csv')
    print('split train and validation')
    train = csv.sample(frac=_TRAIN_RATE)
    validation = csv.drop(train.index)
    print('write to train.tfrecord')
    with tf.python_io.TFRecordWriter('./dataset/train.tfrecord') as train_writer:
        writeToFile(train_writer, train)
    print('write to validation.tfrecord')
    with tf.python_io.TFRecordWriter('./dataset/validation.tfrecord') as validation_writer:
        writeToFile(validation_writer, validation)
    filenames = os.listdir('./dataset/test')
    print('write to predict.tfrecord')
    with tf.python_io.TFRecordWriter('./dataset/predict.tfrecord') as predict_writer:
        for filename in filenames:
            example = make_example('./dataset/test/' + filename, 0)
            predict_writer.write(example.SerializeToString())
    print('write predict filenames')
    pickle.dump(filenames, open('./dataset/predict_filenames.pickle', 'wb'))
    print('done')

if __name__ == '__main__':
    tf.app.run()
