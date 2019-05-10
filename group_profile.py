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
        item_header: Dict[str,int],矩阵的列名
        user_header: Dict[str,int],矩阵行名
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
        self.item_header = ''  # type: Dict[str,int]
        self.user_header = ''  # type: Dict[str,int]
        self.matrix = ''  # type: List[List[float]]
        self.build()

    def build(self, ) -> None:
        """
        构建群体用户的评分矩阵
        构建矩阵的行表头和列表头
        
        Args:
            None
    
        Returns：
            None

        Raises：
            IOError: 
        """

        # 计算矩阵的列名，行名
        user_set, item_set = set(), set()  # type:Set[str]
        for user in self.users:
            if user not in self.data.tr_dict:
                self.u_no_rate.add(user)
            else:
                user_set.add(user)
                for item in self.data.tr_dict[user].keys():
                    item_set.add(item)
        item_list, user_list = list(item_set), list(user_set)
        self.item_header = {item_list[i]: i for i in range(len(item_list))}
        self.user_header = {user_list[i]: i for i in range(len(user_set))}

        # 生成矩阵
        row, col = len(self.user_header), len(self.item_header)
        self.matrix = [[0 for _ in range(col)] for _ in range(row)]
        for user in self.user_header:
            row_index = self.user_header[user]
            for item, score in self.data.tr_dict[user].items():
                col_index = self.item_header[item]
                self.matrix[row_index][col_index] = score
        return

    def gen_profile(self, ) -> List[float]:
        """
        生成群体模型

        Args:
            users:一个群体中的所有用户 id 号
            case:选择用于构建群体抽象的方法
        Returns：
            profile: List[float]
        
        Raises：
            IOError: 
        """

        return None

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
        col_num = np_matrix.shape[1]  # type:int,矩阵的列数
        random_select = 2  # type:随机选取列的个数
        for com in combinations(range(col_num), random_select):
            yield np_matrix[:, com]
        return

    def gen_representative(self) -> None:
        pass


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
    group_file = GroupProfile(['1', '3', '5', '23', '276'], data)
    matrix = group_file.gen_column_coms()
    sub = next(matrix)
    print(sub)
    print(0.0 in sub[0])