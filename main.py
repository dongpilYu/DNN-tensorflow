#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import time
import os
import csv
# from dnn import Dnn
from dnn_CV import Dnn
from sklearn.model_selection import KFold
# 상황에 맞게 둘 중 하나만 로드하면 된다. 

DEVICE = 'GPU'
EPOCH = 5000
BATCH_SIZE = 100
MODEL_PATH = './DNN_Models/model'
HNODES_NUM = [1024,1024,512]
N_SPLITS = 5

if __name__ == '__main__':
     
    # 데이터셋의 위치를 정의하고, 해당 디렉토리의 모든 파일에 순차적으로 접근하여 모델을 생성한다. 
    for (path, dir,files) in os.walk("/home/dong/181112/sol10000/real-valued-20"):
        for filename in files:
            my_dnn = Dnn()
            my_dnn.set_device(DEVICE)
            my_dnn.set_epoch(EPOCH)
            my_dnn.set_batch_size(BATCH_SIZE)
            dataset = my_dnn.load_dataset(path + '/' + filename, shuffle=True)
            # my_dnn.load_datset이 반환 값이 있는 것이 적절한지 살펴보자
            features = np.array(dataset.features)
            labels = np.array(dataset.labels)
            # Cross Validation을 사용해야 하는 상황의 코드
            costs = []
            MODEL_PATH = './DNN_Models'

            start_time = time.time()
            kf = KFold(n_splits=N_SPLITS)
            
            for train_index, test_index in kf.split(features):
                features_train, features_test = features[train_index], features[test_index]
                labels_train, labels_test = labels[train_index], labels[test_index]
                
                my_dnn.load_train_dataset((features_train,labels_train))
                my_dnn.load_test_dataset((features_test,labels_test))
                # load_dataset과는 다르게, 경로가 아닌 배열을 인자로 받는다.  
                my_dnn.train(hnodes_num=HNODES_NUM, model_save_path=MODEL_PATH + '/' + filename + "/model")
                result = my_dnn.eval(model_path=MODEL_PATH + '/' + filename + "/model")
                costs.append(result['cost'])

            end_time = time.time()
            sumOfCost = 0.0
            for cost in costs:
                sumOfCost = sumOfCost + cost
            sumOfCost = sumOfCost / N_SPLITS

            f = open('result_file', 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow([filename, end_time-start_time, sumOfCost])
            f.close()
            # Cross Validation을 사용하지 않는 상황의 코드
            # 따로 test_dataset을 로드하지 않는다면, 학습 데이터를 그대로 테스트 데이터로 사용
            """
            cost = 0 
            MODEL_PATH = './DNN_Models'
            start_time = time.time()

            my_dnn.train(hnodes_num=HNODES_NUM, model_save_path=MODEL_PATH + "/" + filename + "/model")
            result = my_dnn.eval(model_path=MODEL_PATH + "/" + filename + "/model")
            cost = result['cost']
            end_time = time.time()

            f = open('result_file', 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow([filename, end_time-start_time, cost])
            f.close()
            """
