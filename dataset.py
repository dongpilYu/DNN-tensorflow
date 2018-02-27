#-*- coding: utf-8 -*-

from sklearn.model_selection import KFold

class Dataset:
    """ 데이터셋 클래스 """

    def __init__(self):
        """ 데이터셋 생성자 """
        self._idx = 0
        self.num_examples = 0
        self.features = list()
        self.labels = list()
    
        """ CV를 위한 부분 """
        """
        self._train_num_examples = 0
        self._train_features = list()
        self._train_labels = list()

        self._test_num_examples = 0
        self._test_features = list()
        self._test_labels = list()
        """
    def __del__(self):
        """ 데이터셋 소멸자 """
        
        del self._idx
        del self.num_examples
        del self.features
        del self.labels
        
        """ CV를 위한 부분 """
        """
        del self._train_idx
        del self._train_num_examples
        del self._train_features
        del self._train_labels

        del self._test_idx
        del self._test_num_examples
        del self._test_features
        del self._test_labels
        """
    def reset_batch(self):
        """ 배치 인덱스를 초기화 """
        self._idx = 0

    def next_batch(self, batch_size):
        """
        배치 크기만큼 다음 배치를 반환

        :param batch_size: int
            배치 크기

        :return: tuple
            반환된 배치
        """
        if self._idx >= self.num_examples:
            self.reset_batch()
        print("dongpil")
        print(self.num_examples)
        i = self._idx
        j = self._idx + batch_size
        self._idx += batch_size
        print(self.features[i:j])
        return self.features[i:j], self.labels[i:j]
