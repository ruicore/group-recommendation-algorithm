# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:11:29
# @Last Modified by:   何睿
# @Last Modified time: 2019-05-09 14:35:52

import os
import csv
import codecs
import random
import itertools
import collections


class Data(object):
    """
    准备数据集合，为输入的评分记录建立对象
    以用户为键，用户评分过的所有物品为为值构建字典
    附加一些对数据操作的功能

    Attributes:
        tr_dict: dict, {user1:{item1:score1,item2:score2},user2...}
            所有类都将使用的训练数据
        te_dict: dict, {user1:{item1:score1,item2:score2},user2...}
            所有类都将使用的测试数据
        tr_user: list, [user1,user2...] 训练集中的所有用户
        tr_item: set, {item1,item2...} 训练集中的所有物品
        te_user: list, [user1,user2...] 测试集中所有的用户
        te_item: set, {item1,item2...} 测试集中所有的物品
        tr_user_com_items: 
            dict, {user1:{user2:{item1,item2},}} 两个用户共同评价过的物品
        tr_average: dict,{user1:average1,user2:} 训练集中每个用户对所有项目的平均评分
    """

    def __init__(self, tr_data: [list], te_data: [list]):
        """
        建立对象

        Args:
            tr_data: 训练数据 list contains lists，["userId", "movieId", "rating", "timestamp"]
            te_data: 测试数据 list contains lists，["userId", "movieId", "rating", "timestamp"]
    
        """

        self.tr_dict = dict()
        self.te_dict = dict()
        self.tr_user = list()
        self.tr_item = set()
        self.te_user = list()
        self.te_item = set()
        self.tr_user_com_items = dict()
        self.build(tr_data, self.tr_dict, self.tr_user, self.tr_item)
        self.build(te_data, self.te_dict, self.te_user, self.te_item)
        self.user_common_items()

    def build(self, data: list, table: dict, user_list: list,
              item_set: set) -> None:
        """
        构建 用户-项目 评分表
        构建所有的用户表
        构建所有的物品表

        Args:
            data:list contains lists，["userId", "movieId", "rating", "timestamp"]
            table: dict,需要构建的字典对象
            user_list: list, 存储所有的用户
            item_set: set,存储所有的物品
    
        Returns：
            None
        
        Raises：
            IOError: 
        """

        userId, movieId, ratingId = range(3)
        for line in data:
            user, item, rating = line[userId], line[movieId], line[ratingId]
            if user not in table:
                table[user] = dict()
                user_list.append(user)
            else:
                table[user][item] = rating
            item_set.add(item)
        return

    def user_common_items(self) -> None:
        """
        构建 tr_user_com_items 表,此表用于存储两个用户共同评价过的项目索引
        """

        # 建立 item -- users 关系表
        # item_users：{item1:{user1,user2...}...}
        item_users = dict()
        for user, items in self.tr_dict.items():
            for item in items.keys():
                if item not in item_users: item_users[item] = set()
                item_users[item].add(user)

        for item, users in item_users.items():  # users 是对某一个 item 评过分的所有用户
            for com in itertools.combinations(users, 2):
                com = list(com)
                com.sort()
                u, c = com[0], com[1]  # 建立字典，用 u 作为外层主键，c 作为内层主键
                if u not in self.tr_user_com_items:
                    self.tr_user_com_items[u] = dict()
                if c not in self.tr_user_com_items[u]:
                    self.tr_user_com_items[u][c] = set()
                self.tr_user_com_items[u][c].add(item)
        return
    def get_com_items(self,user1:str,user2:str)->list:
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