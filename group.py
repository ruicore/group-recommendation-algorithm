# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:09:59
# @Last Modified by:   何睿
# @Last Modified time: 2019-05-09 22:43:23

import os
import csv
import math
import numpy  # type: ignore
import random
import codecs
import collections
from pprint import pprint
from decimal import Decimal
from itertools import combinations
from typing import List, Dict, Set, Tuple, Callable, Generator, NoReturn
from dataset import Data


class GroupProfile(object):
    """
    使用不同的方式生成群体 Profile
    
    Attributes:
        item_header: Dict[str,int], 矩阵的列名, item : 列号，无序
        user_header: Dict[str,int], 矩阵行名，user：行号，无序
        item_list: List, 矩阵的列名, 有序, 当前群体已经评价过的所有物品
        user_list: List, 矩阵的行名, 有序，users中至少对一项物品有过评价的所有成员
        lm_profile: List, 使用 least misery strategy 生成群体 profile
        avg_profile: List, 使用 average strategy 生成群体 profile
        am_profile: List, 使用 average without misery 生成群体 profile
        mcs_profile: List, 使用 member contribution score 生成群体 profile
        
    """

    def __init__(self, users: List[str], data: Callable) -> None:
        """
        建立对象

        Args:
            users: 一个群体中的所有成员
            data: 数据对象的引用

        Attributes:
            __data: 引用, 对已经对象化的数据对象的引用
            __users：List,一个群体的所有用户
            __u_no_rate：Set, 存放没有任何评分记录的用户，无法对此类用户推荐
            __matrix: List[List[float]], 评分矩阵，用 0 填充未知项
    
        """

        self.__data = data  # type：Callable
        self.__users = users  # type: List[str]
        self.__u_no_rate = set()  # type: Set[str]
        self.item_list = list()  # type:List[str]
        self.user_list = list()  # type:List[str]
        self.item_header = dict()  # type: Dict[str,int]
        self.user_header = dict()  # type: Dict[str,int]
        self.lm_profile = list()  # type:List[float]
        self.avg_profile = list()  # type:List[float]
        self.am_profile = list()  # type:List[float]
        self.mcs_profile = list()  # type:List[float]
        self.__matrix = list()  # type: List[List[float]]

        self.__build()
        self.lm_profile = self.__gen_lm_profile()
        self.avg_profile = self.__gen_avg_profile()
        self.am_profile = self.__gen_am_profile()
        self.mcs_profile = self.__gen_mcs_profile()

    def __build(self) -> None:
        """
        构建群体用户的评分矩阵，用 0 填充未评分项目
        
        Args:
            None
    
        Returns：
            NoReturn

        Raises：

        """

        # 计算矩阵的列名，行名
        user_set, item_set = set(), set()  # type:Set[str],Set[str]
        for user in self.__users:
            if user not in self.__data.tr_dict:
                self.__u_no_rate.add(user)
            else:
                user_set.add(user)
                for item in self.__data.tr_dict[user].keys():
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
        self.__matrix = [[0 for _ in range(col)] for _ in range(row)]

        for user, row_index in self.user_header.items():
            for item, score in self.__data.tr_dict[user].items():
                col_index = self.item_header[item]
                self.__matrix[row_index][col_index] = score

        return

    def __gen_column_coms(self, ) -> Generator:
        """
        求矩阵列的两两组合

        Args:
            None
    
        Returns：
            Generator:sub_matrix,对原矩阵求列的两两组合
    
        Raises：
            IOError: 
        """

        np_matrix = numpy.array(self.__matrix)
        col_num = np_matrix.shape[1]  # type:int
        random_select = 2  # type:int

        for com in combinations(range(col_num), random_select):
            yield np_matrix[:, com]

        return

    def __gen_repre(self, matrix: List[List[float]]) -> List[Tuple[str, int]]:
        """
        求矩阵中的代表成员

        Args:
            matrix,评分矩阵
    
        Returns：
            repre_users:代表性成员
    
        Raises：
            IOError: 
        """

        # 排除有为评分记录的成员
        exclude = 0  # type:int
        # 记录有完整评分记录的用户
        user_list = list()  # type:List[str]
        # 有完整评分记录用户的评分矩阵
        rated = list()  # type:List[List[float]]

        # 排除有未评分记录的 user
        for index, vector in enumerate(matrix):
            if exclude not in vector:
                user_list.append(self.user_list[index])
                rated.append(vector)

        # 计算相似度
        repre_users = dict()  # type:Dict
        if len(user_list) == 0: return list(tuple())  # 没有用户返回空

        avg_vector = numpy.mean(rated, axis=0)  # 行向量为一个整体，求平均值
        for index, row in enumerate(rated):
            vector = numpy.array(row)

            # 计算余弦相似度
            num = numpy.dot(vector, avg_vector)
            denom = numpy.linalg.norm(vector) * numpy.linalg.norm(avg_vector)
            cosin = num / denom

            repre_users[user_list[index]] = cosin

        # 当数据量不大时，全部返回
        return sorted(repre_users.items(), key=lambda x: x[1])

    def __gen_lm_profile(self) -> List[float]:
        """
        使用 least misery 策略，生成群体 profile

        Args:
            None

        Returns：
            profile: 群体的特征，即群体对物品的评分

        Raises：
            IOError: 
        """
        # 求每列大于 0 的最小值

        profile = list()
        np_matrix = numpy.array(self.__matrix)

        for i in range(len(np_matrix[0])):
            vector = np_matrix[:, i]
            # vector 一定存在一个大于 0 的数
            num = min(x for x in vector if x > 0)
            profile.append(num)

        return profile

    def __gen_avg_profile(self) -> List[float]:
        """
        使用 average 策略，生成群体 profile

        Args:
            None

        Returns：
            profile: 群体的特征，即群体对物品的评分
            
        Raises：
            IOError: 
        """
        # 求每列大于 0 的均值
        profile = self.__gen_am_profile(T=0)

        return profile

    def __gen_am_profile(self, T: float = 2) -> List[float]:
        """
        使用 average without misery 策略，生成群体 profile

        Args:
            T: flaot
                threshshold to filter out items that will cause disappointment
                for members who have ratings lower than T，default is set to 2
        Returns：
            profile: 群体的特征，即群体对物品的评分
            
        Raises：
            IOError: 
        """

        # 求每列大于 T 的所有数均值
        profile = list()
        np_matrix = numpy.array(self.__matrix)

        for i in range(len(np_matrix[0])):
            vector = np_matrix[:, i]
            a = sum(x for x in vector if x > T)
            b = sum(1 for x in vector if x > T)
            if b != 0:
                profile.append(a / b)
            else:
                profile.append(max(x for x in vector))

        return profile

    def __gen_mcs_profile(self) -> List[float]:
        """
        使用成员贡献分数, 生成群体 profile

        Args:
            None
        Returns：
            profile: 群体的特征，即群体对物品的评分
        
        Raises：
            IOError: 
        """
        profile = list()
        user_weight = dict()  # type:Dict[str,float]

        item_cout = len(self.item_list)  # type:int
        # user_cout = len(self.user_list)

        # 统计每个成员作为代表性成员的次数

        for matrix in self.__gen_column_coms():
            for user, _ in self.__gen_repre(matrix):
                user_weight[user] = user_weight.get(user, 0) + 1

        # 计算每个成员的代表分
        for user in user_weight:
            user_weight[user] *= 2 / item_cout

        # 计算每个成员在群体中的权重
        weight_sum = sum(user_weight.values())
        for user in user_weight:
            user_weight[user] /= weight_sum

        # 计算该群体对每件物品的评分

        for col, item in enumerate(self.item_list):
            # 有过评分记录的用户
            rated_users = [
                u for row, u in enumerate(self.user_list)
                if self.__matrix[row][col] != 0
            ]

            rating = 0.0
            # 每个用户的权重归一化
            _max_weight = max(user_weight[u] for u in rated_users)

            for user in rated_users:
                score = self.__data.tr_dict[user][item]
                rating += user_weight[user] / _max_weight * score

            rating = float(Decimal(rating).quantize(Decimal("0.00")))
            profile.append(rating)

        return profile