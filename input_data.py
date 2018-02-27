#-*- coding: utf-8 -*-

import csv
import os
import random

from dataset import Dataset

random.seed(486)


def load_dataset(path, shuffle=False):
    """
    데이터셋 파일을 읽고, 학습 가능한 형태로 반환.

    :param path: str
        데이터셋 파일의 경로
    :param shuffle: boolean
        데이터 순서 섞기 여부

    :return: class
        데이터셋 클래스
    """
    if not os.path.exists(path):
        print("The path '{}' does not exist.".format(path))
        raise FileNotFoundError
    else:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            # 컬럼명
            for _ in reader:
                break

            # 데이터
            data_list = list()
            for row in reader:
                feature = [float(r) for r in row[0:-1]]
                label = [float(row[-1])]
                data_list.append((feature, label))

        if shuffle:
            random.shuffle(data_list)

        dataset = Dataset()
        dataset.num_examples = len(data_list)
        for data in data_list:
            dataset.features.append(data[0].copy())
            dataset.labels.append(data[1])
        print('The dataset has been successfully loaded from: {}'.format(path))
        return dataset

def load_train_dataset(dataset, train, shuffle=False):
    dataset.num_examples = len(train[1])
    dataset.features = train[0]
    dataset.labels = train[1]
    print('The train dataset has been successfully loaded')
    return dataset

def load_test_dataset(dataset, test, shuffle=False):
    dataset.num_examples = len(test[1]) 
    dataset.features = test[0]
    dataset.labels = test[1]
    print('The test dataset has been successfully loaded')
    return dataset

