from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf  # pylint: disable=E0012

import resnet_model
import resnet_run_loop

_ORIGIN_SIZE = 350
_IMAGE_SIZE = 250
_NUM_CHANNELS = 3
_NUM_CLASSES = 8
_NUM_IMAGES = {
    'train': 0,
    'validation': 0,
    'predict': 0
}


def preprocess_image(image_buffer, is_training, n_crop=None):
    image = tf.reshape(tf.image.decode_jpeg(
        image_buffer), [_ORIGIN_SIZE, _ORIGIN_SIZE, _NUM_CHANNELS])
    # 从350x350的图片的靠下部分剪裁出一个300x300的小图，因为图片的上部基本是头
    image = tf.image.crop_to_bounding_box(image, 50, 25, 300, 300)

    if is_training:
        # 从小图中随机剪裁输入
        image = tf.random_crop(
            image, [_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)          # 亮度
        image = tf.image.random_contrast(image, 0.8, 1.2)       # 对比度
        image = tf.image.random_hue(image, 0.02)                # 色调
        image = tf.image.random_saturation(image, 0.8, 1.2)     # 饱和度
    elif n_crop is None:
        # 中心剪裁小图作为输入
        image = tf.image.crop_to_bounding_box(image, 25, 25, 250, 250)
    else:
        # 小图中10-crop
        ten_crop = [
            [0, 0, 250, 250],
            [50, 0, 250, 250],
            [0, 50, 250, 250],
            [50, 50, 250, 250],
            [25, 25, 250, 250]
        ]
        bbox = ten_crop[int(n_crop / 2)]
        image = tf.image.crop_to_bounding_box(image, *bbox)
        if n_crop % 2 == 1:
            image = tf.image.flip_left_right(image)

    image = tf.cast(image, tf.float32)
    return image


def parse_record(raw_record, is_training, n_crop=None):
    feature_map = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }
    features = tf.parse_single_example(raw_record, feature_map)
    image_buffer = features['image']
    label = tf.one_hot(features['label'], _NUM_CLASSES)
    image = preprocess_image(image_buffer, is_training, n_crop=n_crop)
    return image, label


def input_fn(mode, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1, multi_gpu=False, n_crop=None):
    filename = os.path.join(data_dir, mode+'.tfrecord')
    dataset = tf.data.TFRecordDataset(
        [filename], num_parallel_reads=num_parallel_calls)

    num_images = _NUM_IMAGES[mode]

    return resnet_run_loop.process_record_dataset(
        dataset, mode == 'train', batch_size, num_images, parse_record,
        num_epochs, num_parallel_calls, examples_per_epoch=num_images,
        multi_gpu=multi_gpu, n_crop=n_crop)


def get_synth_input_fn():
    return resnet_run_loop.get_synth_input_fn(
        _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES)


class Model(resnet_model.Model):
    def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
                 version=resnet_model.DEFAULT_VERSION):
        if resnet_size < 50:
            bottleneck = False
            final_size = 512
        else:
            bottleneck = True
            final_size = 2048

        super(Model, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            final_size=final_size,
            version=version,
            data_format=data_format)


def _get_block_sizes(resnet_size):
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
                   resnet_size, choices.keys()))
        raise ValueError(err)


def model_fn(features, labels, mode, params):
    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

    return resnet_run_loop.resnet_model_fn(features, labels, mode, Model,
                                           resnet_size=params['resnet_size'],
                                           weight_decay=1e-4,
                                           learning_rate_fn=learning_rate_fn,
                                           momentum=0.9,
                                           data_format=params['data_format'],
                                           version=params['version'],
                                           loss_filter_fn=None,
                                           multi_gpu=params['multi_gpu'])


def main(argv):
    parser = resnet_run_loop.ResnetArgParser(
        resnet_size_choices=[18, 34, 50, 101, 152, 200])

    parser.set_defaults(
        train_epochs=100,
        data_dir='../dataset',
        model_dir='./model'
    )

    flags = parser.parse_args(args=argv[1:])

    train_path = os.path.join(flags.data_dir, 'train.tfrecord')
    validation_path = os.path.join(flags.data_dir, 'validation.tfrecord')
    predict_path = os.path.join(flags.data_dir, 'predict.tfrecord')
    _NUM_IMAGES['train'] = sum(
        1 for _ in tf.python_io.tf_record_iterator(train_path))
    _NUM_IMAGES['validation'] = sum(
        1 for _ in tf.python_io.tf_record_iterator(validation_path))
    _NUM_IMAGES['predict'] = sum(
        1 for _ in tf.python_io.tf_record_iterator(predict_path))

    input_function = flags.use_synthetic_data and get_synth_input_fn() or input_fn

    resnet_run_loop.resnet_main(
        flags, model_fn, input_function,
        shape=[_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
