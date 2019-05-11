# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:09:59
# @Last Modified by:   何睿
# @Last Modified time: 2019-05-09 22:43:23

import os
import csv
import numpy  # type: ignore
import random
import codecs
import decimal
import collections
from pprint import pprint
from decimal import Decimal
from itertools import combinations
from typing import List, Dict, Set, Tuple, Callable, Generator
from dataset import Data


class GroupProfile(object):
    """
    生成群体 Profile
    
    Attributes:
        data: 引用，对已经对象化的数据对象的引用
        users：List，一个群体的所有用户
        u_no_rate：Set,存放没有任何评分记录的用户，无法对此类用户推荐
        item_header: Dict[str,int],矩阵的列名, item : 列号，无序
        user_header: Dict[str,int],矩阵行名，user：行号，无序
        item_list: List,矩阵的列名，有序, 当前群体已经评价过的所有物品
        user_list: List,矩阵的行名，有序，users中至少对一项物品有过评价的所有成员
        matrix：List[List[float]],评分矩阵，用 0 填充未知项
    """

    def __init__(self, users: List[str], data: Callable):
        """
        建立对象

        Args:
            users: 群体成员数组
            data: 数据对象的引用
    
        """

        self.data = data
        self.users = users  # type: List[str]
        self.u_no_rate = set()  # type: Set[str]
        self.item_list = list()  # type:List[str]
        self.user_list = list()  # type:List[str]
        self.item_header = dict()  # type: Dict[str,int]
        self.user_header = dict()  # type: Dict[str,int]
        self.matrix = list()  # type: List[List[float]]
        self.build()

    def build(self, ) -> None:
        """
        构建群体用户的评分矩阵，用 0 填充未评分项目
        
        Args:
            None
    
        Returns：
            None

        Raises：

        """

        # 计算矩阵的列名，行名
        user_set, item_set = set(), set()  # type:Set[str],Set[str]
        for user in self.users:
            if user not in self.data.tr_dict:
                self.u_no_rate.add(user)
            else:
                user_set.add(user)
                for item in self.data.tr_dict[user].keys():
                    item_set.add(item)

        self.item_list, self.user_list = list(item_set), list(user_set)

        self.item_header = {
            self.item_list[i]: i
            for i in range(len(self.item_list))
        }
        self.user_header = {
            self.user_list[i]: i
            for i in range(len(self.user_list))
        }

        # 生成矩阵
        row, col = len(self.user_header), len(self.item_header)
        self.matrix = [[0 for _ in range(col)] for _ in range(row)]

        for user, row_index in self.user_header.items():
            for item, score in self.data.tr_dict[user].items():
                col_index = self.item_header[item]
                self.matrix[row_index][col_index] = score
        return

    def gen_profile(self, ) -> List[float]:
        """
        生成群体模型

        Args:
            None
        Returns：
            profile: List[float]
        
        Raises：
            IOError: 
        """

        user_weight = dict()  # type:Dict[str,float]
        item_cout = len(self.item_list)  # type:int

        for matrix in self.gen_column_coms():
            for user, _ in self.gen_representative(matrix):
                user_weight[user] = user_weight.get(user, 0) + 1

        for user in user_weight:
            user_weight[user] *= 2 / item_cout

        weight_sum = sum(user_weight.values())
        for user in user_weight:
            user_weight[user] /= weight_sum

        profile = []  # type:List[float], 群体模型
        for item in self.item_list:
            rating = 0.0
            for user, weight in user_weight.items():
                row, col = self.user_header[user], self.item_header[item]
                rating += self.matrix[row][col] * weight
            rating = float(Decimal(rating).quantize(Decimal("0.00")))
            profile.append(rating)

        return profile

    def gen_column_coms(self, ) -> Generator:
        """
        求矩阵列的两两组合

        Args:
            None
    
        Returns：
            Generator:sub_matrix,对原矩阵求列的两两组合
    
        Raises：
            IOError: 
        """
        np_matrix = numpy.array(self.matrix)
        col_num = np_matrix.shape[1]  # type:int, 矩阵的列数
        random_select = 2  # type:int, 随机选取列的个数
        for com in combinations(range(col_num), random_select):
            yield np_matrix[:, com]
        return

    def gen_representative(self,
                           matrix: List[List[float]]) -> List[Tuple[str, int]]:
        """
        求矩阵中的代表成员

        Args:
            matrix,评分矩阵
    
        Returns：
            repre_users:代表性成员
    
        Raises：
            IOError: 
        """

        exclude = 0  # type:int,排除有为评分记录的成员
        user_list = list()  # type:List[str], 记录有完整评分记录的用户
        m = []  # type:List[List[float]] ,有完整评分记录用户的评分矩阵

        # 排除有未评分记录的 user
        for index, vector in enumerate(matrix):
            if exclude not in vector:
                user_list.append(self.user_list[index])
                m.append(vector)

        # 计算相似度
        repre_users = dict()  # type:Dict
        if len(user_list) == 0: return tuple()  # 没有用户返回空
        # if len(user_list) == 1: repre_users[user_list[0]] = 1  # 只有一个用户

        avg_vector = numpy.mean(m, axis=0)  # 行向量为一个整体，求平均值
        for index, row in enumerate(m):
            vector = numpy.array(row)
            num = numpy.dot(vector, avg_vector)
            denom = numpy.linalg.norm(vector) * numpy.linalg.norm(avg_vector)
            cosin = num / denom
            repre_users[user_list[index]] = cosin

        # 当数据量不大时，全部返回
        return sorted(repre_users.items(), key=lambda x: x[1], reverse=False)


if __name__ == "__main__":
    rate = 0.5
    movile_path = r"C:\HeRui\Git\GroupRecommenderSystem\movies\movies_small\ratings.csv"
    tran_list, test_list = [], []
    with codecs.open(movile_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 去掉表头
        k = rate * 100
        for row in reader:
            if random.randint(1, 101) <= k: tran_list.append(row)
            else: test_list.append(row)
    data = Data(tran_list, test_list)
    group = GroupProfile(['67', '3', '5', '23', '276'], data)
    matrix = group.gen_column_coms()
    profile = group.gen_profile()