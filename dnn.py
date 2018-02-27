#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import input_data

SEED = 486

tf.set_random_seed(SEED)


class Dnn:
    """
    심층신경망(Deep Neural Networks) 클래스
    """

    def __init__(self):
        """ 클래스 생성자 """
        self.__load_dataset = False
        self.__load_model = False

        self._dataset = None
        self._model = None

        self._dataset_size = None
        self._feature_size = None
        self._label_size = None

        self._device = 'CPU'
        self._epoch = 100
        self._batch_size = 100

    def __del__(self):
        """ 클래스 소멸자 """
        del self.__load_dataset
        del self.__load_model

        del self._dataset
        del self._model

        del self._dataset_size
        del self._feature_size
        del self._label_size

        del self._device
        del self._epoch
        del self._batch_size

    def load_dataset(self, path, shuffle=True):
        """
        특정 경로에서 데이터셋을 로드

        :param path: str
            로드할 데이터셋 파일의 경로
        """
        if not isinstance(path, str):
            raise TypeError("type of 'path' must be str.")

        self.__load_dataset = True
        self._dataset = input_data.load_dataset(path, shuffle=shuffle)
        self._dataset_size = self._dataset.num_examples
        self._feature_size = len(self._dataset.features[0])
        #self._label_size = len(self._dataset.labels[0])
        self._label_size = 1

    def load_model(self, path, sess):
        """
        특정 경로에서 모델을 로드

        :param path: str
            로드할 모델 파일의 경로
        :param sess: tf.Session
            세션
        """
        if not isinstance(path, str):
            raise TypeError("type of 'path' must be str.")

        par_dir = os.path.split(path)[0]
        meta_file = '{}.meta'.format(path)
        ckpt_file = '{}/checkpoint'.format(par_dir)

        if not os.path.exists(meta_file):
            raise FileNotFoundError(
                'The meta file is not exist from: {}'.format(par_dir))

        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(
                'The checkpoint file is not exist from: {}'.format(par_dir))

        self.__load_model = True
        self._model = tf.train.import_meta_graph(meta_file, clear_devices=True)
        self._model.restore(sess, tf.train.latest_checkpoint(par_dir))

    def set_device(self, device='cpu'):
        """
        신경망을 실행하는 Device를 설정.

        :param device: str
            'cpu' 또는 'gpu'
        """
        if not isinstance(device, str):
            raise TypeError("type of 'path' must be str.")

        if device.upper() in ['CPU', 'GPU']:
            self._device = device.upper()
        else:
            raise ValueError("'device' must be 'CPU' or 'GPU'.")

    def set_epoch(self, epoch=100):
        """
        Epoch 크기를 설정.

        :param epoch: int
            설정하고자 하는 epoch 크기
        """
        if not isinstance(epoch, int):
            raise ValueError

        self._epoch = epoch

    def set_batch_size(self, batch_size=100):
        """
        Epoch 크기를 설정.

        :param batch_size: int
            설정하고자 하는 batch 크기
        """
        if not isinstance(batch_size, int):
            raise ValueError

        self._batch_size = batch_size

    def train(self, hnodes_num, dropout_layers=None, keep_prob=1.0, model_save_path='./DNN_Models/model'):
        """
        신경망을 학습하고 모델을 저장.

        :param hnodes_num: list
            은닉계층의 노드 개수
        :param keep_prob: float
        :param dropout_layers: list
            드롭아웃을 실행할 은닉계층의 번호
        :param model_save_path: str
            학습된 모델을 저장할 경로
        :return: (int, list)
        """
        if not isinstance(hnodes_num, list):
            return TypeError("type of 'hnodes_num' must be list.")

        if dropout_layers:
            if not isinstance(dropout_layers, list):
                return TypeError("type of 'dropout_layers' must be list.")
            if max(dropout_layers) > len(hnodes_num):
                return ValueError('')

            dropout_layers.sort()
            dropout_layers = set(dropout_layers)

        if not self.__load_dataset:
            return RuntimeError("Please Load Dataset by 'load_dataset(path).'")

        with tf.device('/device:{}:0'.format(self._device)):
            x = tf.placeholder(dtype=tf.float32, shape=[
                               None, self._feature_size], name='input_x')
            y = tf.placeholder(dtype=tf.float32, shape=[
                               None, self._label_size], name='input_y')

            keep_prob_ = tf.placeholder(dtype=tf.float32, name='keep_prob')
            learning_rate = tf.placeholder(
                dtype=tf.float32, name='learning_rate')

            with tf.variable_scope('Hidden_Layer'):
                hlayers = list()
                for (i, n) in enumerate(hnodes_num):
                    if i == 0:
                        hlayers.append(tf.layers.dense(
                            inputs=x, units=n, activation=tf.nn.leaky_relu, use_bias=True))
                    else:
                        hlayers.append(
                            tf.layers.dense(
                                inputs=hlayers[i - 1], units=n, activation=tf.nn.leaky_relu, use_bias=True)
                        )

                    if dropout_layers:
                        if (i + 1) in dropout_layers:
                            hlayers[i] = tf.nn.dropout(
                                hlayers[i], keep_prob=keep_prob_, seed=SEED)

            y_predict = tf.layers.dense(
                inputs=hlayers[-1], units=self._label_size, activation=None, use_bias=True, name='y_predict'
            )

        with tf.device('/device:CPU:0'):
            cost = tf.reduce_mean(
                tf.losses.mean_squared_error(labels=y, predictions=y_predict))
            train_step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(cost)

            model_saver = tf.train.Saver()

        # set train count
        if self._dataset_size <= self._batch_size:
            train_count = 1
        else:
            if (self._dataset_size % self._batch_size) == 0:
                train_count = int(self._dataset_size / self._batch_size)
            else:
                train_count = int(self._dataset_size / self._batch_size) + 1

        # make directory
        par_dir = os.path.split(model_save_path)[0]
        if not os.path.exists(par_dir):
            os.makedirs(par_dir)

        lr = 0.001

        # run session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # optimization
            for step in range(self._epoch):
                if (step > 0) and (step % 100 == 0):
                    print('STEP: {}'.format(step))
                self._dataset.reset_batch()
                if (step > 0) and (step % 20 == 0):
                    lr *= 0.1
                for i in range(train_count):
                    batch_x, batch_y = self._dataset.next_batch(
                        self._batch_size)
                    sess.run(
                        train_step,
                        feed_dict={x: batch_x, y: batch_y,
                                   keep_prob_: keep_prob, learning_rate: lr}
                    )

            # save model
            model_saver.save(sess, model_save_path)
            print("Model save to '{}'".format(model_save_path))

        tf.reset_default_graph()

    def eval(self, model_path):
        """
        학습된 모델을 평가.

        :param model_path: str
            로드할 모델 파일의 경로

        :return: list
        """
        if not self.__load_dataset:
            raise RuntimeError("Please Load Dataset by 'load_dataset(path).'")

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self.load_model(model_path, sess)
        graph = tf.get_default_graph()

        with tf.device('/{}:0'.format(self._device)):
            # Input
            x = graph.get_tensor_by_name('input_x:0')
            y = graph.get_tensor_by_name('input_y:0')

            # Keep prob
            keep_prob = graph.get_tensor_by_name('keep_prob:0')

            y_predict = graph.get_tensor_by_name('y_predict/BiasAdd:0')

        with tf.device('/device:CPU:0'):
            cost = tf.reduce_mean(
                tf.losses.mean_squared_error(labels=y, predictions=y_predict))

        if self._dataset_size <= self._batch_size:
            test_count = 1
        else:
            if (self._dataset_size % self._batch_size) == 0:
                test_count = int(self._dataset_size / self._batch_size)
            else:
                test_count = int(self._dataset_size / self._batch_size) + 1

        # test
        self._dataset.reset_batch()
        for i in range(test_count):
            batch_x, batch_y = self._dataset.next_batch(self._batch_size)
            cost_ = sess.run(
                cost,
                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}
            )

        tf.reset_default_graph()
        sess.close()

        return {
            'cost': cost_
        }

    def query(self, feature, model_path):
        """
        학습된 신경망에 질의.

        :param weather: list
            특징 데이터
        :param model_path: str
            로드할 모델 파일의 경로

        :return: list
        """
        sess = tf.Session()

        self.load_model(model_path, sess)
        graph = tf.get_default_graph()

        with tf.device('/{}:0'.format(self._device)):
            # Input
            x = graph.get_tensor_by_name('input_x:0')

            # Keep prob
            keep_prob = graph.get_tensor_by_name('keep_prob:0')

            y_predict = graph.get_tensor_by_name('y_predict/BiasAdd:0')
            predict = tf.argmax(y_predict, 1)

            # run session
            result = sess.run(predict, feed_dict={x: feature, keep_prob: 1.0})

        sess.close()

        return result
