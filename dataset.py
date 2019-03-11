# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:11:29
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 15:08:08

import os
import codecs
import random


class Movies:
    def __init__(self):
        # 文件根路径
        self.base = os.path.abspath(".")

    def spilt_data(self, M: int, k: int, seed: int) -> dict:
        """
        将数据集随机分成训练集和测试集

        Args:
            M: int，数据集分成的总份数，一份为测试集，M-1份为训练集
            k: int，0<=k<=M-1,选取一份作为测试集
            seed: int,随机数种子
            train,test :dict, {user1:{item1:count,item2:count},user2...}

        Returns:
            train:dict，训练集
            test：dict，测试集
        """

        train_file = os.path.join(self.base, 'movies/u1.base')
        test_file = os.path.join(self.base, 'movies/u1.test')

        with codecs.open(train_file, 'r') as f:
            train_data = [line.split()[:2] for line in f.readlines()]
        with codecs.open(test_file, 'r') as f:
            test_data = [line.split()[:2] for line in f.readlines()]

        test, train = dict(), dict()
        for user, item in train_data:
            if user not in train: train[user] = dict()
            train[user][item] = train[user].get(item, 0) + 1
        for user, item in test_data:
            if user not in test: test[user] = dict()
            test[user][item] = test[user].get(item, 0) + 1

        # random.seed(seed)
        # for user, item in data:
        #     if random.randint(0, M) == k:
        #         if user not in test: test[user] = dict()
        #         test[user][item] = test[user].get(item, 0) + 1
        #     else:
        #         if user not in train: train[user] = dict()
        #         train[user][item] = train[user].get(item, 0) + 1
        return train, test


if __name__ == "__main__":
    movie = Movies()
    train, test = movie.spilt_data(8, 2, 1)
    print(train)
