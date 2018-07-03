import tensorflow as tf
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000,
                    type=int, help='number of training steps')
parser.add_argument('--model_dir', default='./model/linear_regression',
                    type=str, help='path to save model')
parser.add_argument('--output_path', default='./linear_regression_output.csv',
                    type=str, help='path to save output csv')


def main(argv):
    args = parser.parse_args(argv[1:])
    data_dir = os.path.abspath('./dataset')
    output_path = os.path.abspath(args.output_path)

    (train_x, train_y), test_x = load_data(data_dir)

    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    model = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        model_dir=args.model_dir
    )

    model.train(
        input_fn=lambda: input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps
    )

    predictions = model.predict(
        input_fn=lambda: input_fn(test_x, None, args.batch_size, False)
    )
    result = []
    for prediction in predictions:
        result.append(prediction['predictions'][0])
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
    train_x, train_y = train, train.pop('reference')

    test_x = pd.read_csv(open(test_path))
    test_x.pop('id')

    return (train_x, train_y), test_x


def input_fn(features, labels, batch_size, training=True):
    features = dict(features)
    if (labels is None):
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if (training):
        dataset = dataset.shuffle(batch_size * 10).repeat()
    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
