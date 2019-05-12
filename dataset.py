# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:11:29
# @Last Modified by:   何睿
# @Last Modified time: 2019-05-09 14:35:52

import os
import csv
import numpy  # type: ignore
import random
import codecs
import collections
from pprint import pprint
from decimal import Decimal
from itertools import combinations
from typing import List, Dict, Set, Tuple


class Data(object):
    """
    准备数据集合，为输入的评分记录建立对象
    以用户为键，用户评分过的所有物品为值构建字典
    附加一些对数据操作的功能

    Attributes:
        tr_dict: dict, {user1:{item1:score1,item2:score2},user2...}
            所有类都将使用的训练数据
        te_dict: dict, {user1:{item1:score1,item2:score2},user2...}
            所有类都将使用的测试数据
        tr_user: list, [user1,user2...] 
            训练集中的所有用户
        tr_item: set, {item1,item2...} 
            训练集中的所有物品
        te_user: list, [user1,user2...] 
            测试集中所有的用户
        te_item: set, {item1,item2...} 
            测试集中所有的物品
        tr_item_com_users: 
            dict, {item1:{item2:{user1,user2...},}} 
            评价过两个物品的公共 user
        tr_average: dict,{user1:average1,user2:} 
            训练集中每个用户对所有项目的平均评分
    """

    def __init__(self, tr_data: List[List[str]], te_data: List[List[str]]):
        """
        建立对象

        Args:
            tr_data: 训练数据, ["userId", "movieId", "rating", "timestamp"]
            te_data: 测试数据, ["userId", "movieId", "rating", "timestamp"]
    
        """

        self.tr_dict = dict()  # type: Dict[str, Dict[str,float]]
        self.te_dict = dict()  # type: Dict[str, Dict[str,float]]
        self.tr_user = list()  # type: List[str]
        self.tr_item = set()  # type: Set[str]
        self.te_user = list()  # type: List[str]
        self.te_item = set()  # type: Set[str]
        self.tr_average = dict()  # type: Dict[str, float]
        self.tr_item_com_users = dict()  # type: Dict[str, Dict[str, Set[str]]]
        self.__build(tr_data, self.tr_dict, self.tr_user, self.tr_item)
        self.__build(te_data, self.te_dict, self.te_user, self.te_item)
        self.__build_item_common_users()
        self.__build_average()

    def __build(self, data: List[List[str]], table: Dict, user_list: List[str],
                item_set: Set[str]) -> None:
        """
        构建 用户-项目 评分表
        构建所有的用户表
        构建所有的物品表

        Args:
            data: ["userId", "movieId", "rating", "timestamp"]
            table: 需要构建的字典对象
            user_list: 存储所有的用户
            item_set: 存储所有的物品
    
        Returns：
            None
        
        Raises：
            IOError: 
        """

        uId, mId, rId = range(3)  # 用户 id ，电影 id，评分 id 分别对应的索引
        for line in data:
            user, item, rating = line[uId], line[mId], float(line[rId])
            if user not in table:
                table[user] = dict()
                user_list.append(user)
            else:
                table[user][item] = rating
            item_set.add(item)
        return

    def __build_item_common_users(self) -> None:
        """
        构建 tr_item_com_users 表,此表用于存储评价过两个项目的公共用户索引
        """

        for user, items in self.tr_dict.items():
            for com in combinations(items.keys(), 2):
                item_sorted = list(com)
                item_sorted.sort()

                # 建立字典，用 item1 作为外层主键，item2 作为内层主键
                item1, item2 = item_sorted[0], item_sorted[1]
                if item1 not in self.tr_item_com_users:
                    self.tr_item_com_users[item1] = dict()
                if item2 not in self.tr_item_com_users[item1]:
                    self.tr_item_com_users[item1][item2] = set()
                self.tr_item_com_users[item1][item2].add(user)

        return

    def __build_average(self) -> None:
        """
        计算用户对评价过的所有物品的评分平均数
        """

        for user, items in self.tr_dict.items():
            sum_score, count = sum(items.values()), len(items)
            num = Decimal(sum_score / count).quantize(Decimal("0.00"))
            self.tr_average[user] = float(num)
        return

    def get_com_users(self, item1: str, item2: str) -> List[str]:
        """
        返回评价过两个物品的用户交集  

        Args:
            item1: str，物品 ID 号
            item2: str，物品 ID 号
    
        Returns：
            coms：list，评价过 item1 和用户 item2 的公共用户

        Raises：
            IOError: 
        """
        items = sorted([item1, item2])
        it1, it2 = items[0], items[1]
        commons = []  # type: List[str]
        if it1 in self.tr_item_com_users and it2 in self.tr_item_com_users[it1]:
            commons = list(self.tr_item_com_users[it1][it2])
        return commons


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