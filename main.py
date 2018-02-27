#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import time
import os
import csv
from dnn import Dnn
from sklearn.model_selection import KFold

DEVICE = 'GPU'
EPOCH = 5000
BATCH_SIZE = 100
MODEL_PATH = './DNN_Models/model'
HNODES_NUM = [1024,1024,512]
N_SPLITS = 5

if __name__ == '__main__':
   
    for (path, dir,files) in os.walk("/home/dong/dong/over20/"):
        for filename in files:
            my_dnn = Dnn()
            my_dnn.set_device(DEVICE)
            my_dnn.set_epoch(EPOCH)
            my_dnn.set_batch_size(BATCH_SIZE)
            dataset = my_dnn.load_dataset(path + '/' + filename, shuffle=True)
            
            features = np.array(dataset.features)
            labels = np.array(dataset.labels)
            costs = []

            start_time = time.time()
            kf = KFold(n_splits=N_SPLITS)
            
            for train_index, test_index in kf.split(features):
                features_train, features_test = features[train_index], features[test_index]
                labels_train, labels_test = labels[train_index], labels[test_index]
                
                my_dnn.load_train_dataset((features_train,labels_train))
                my_dnn.load_test_dataset((features_test,labels_test))
                
                my_dnn.train(hnodes_num=HNODES_NUM, model_save_path=MODEL_PATH)
                result = my_dnn.eval(model_path=MODEL_PATH)
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
