# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:10:49
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 16:44:57

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
from typing import List, Dict, Set, Tuple, Callable, Generator
from dataset import Data
from group_profile import GroupProfile


class MCSRecommend(object):
    """
    使用基于成员贡献分数的群体推荐方式

    Attributes:
        users：List，一个群体的所有用户
        data: 引用, 对已经对象化的数据对象的引用
        rated_items: List, 群体中评过分的物品集合
        rated_users: List, users 中至少对一项物品有过评价的所有成员
        profile:List,有序 群体抽象后对物品的评分
        item_score: Dict,群体抽象后对物品的评分，键为物品 ID，值为评分
        non_group_items：Dict,
            没有被此群体评过分的物品，即候选推荐物品集合,j键为物品，值为预测评分
        sim_non_group_members: Dict,
            不在此群体中的其他成员,键为成员，值为该成员与 pseudo user 的相似度
    """

    def __init__(self, ):
        """
        建立对象

        Args:
            users: 群体成员数组
            data: 数据对象的引用
    
        """
        self.users = list()  #type:List[str]
        self.data = None  # type:Callable
        self.profile = list()  # type:List[float]
        self.item_score = dict()  # type:Dict[str,float]
        self.rated_items = list()  # List: Set[str]
        self.rated_users = list()  # List: Set[str]
        self.non_group_items = dict()  # type: Dict[str,float]
        self.sim_non_group_members = dict()  # type: Dict[str,float]

    def build(self, users: List[str], data: Callable) -> None:
        """
        为对象填充数据
        获得群体抽象为个体的特征

        Args:
            None
    
        Returns：
            None

        Raises：

        """
        self.users = users
        self.data = data
        group = GroupProfile(self.users, self.data)
        self.profile = group.gen_profile()
        self.rated_items = group.item_list
        self.rated_users = group.user_list
        assert len(self.profile) == len(self.rated_items)
        self.item_score = {
            item: score
            for item, score in zip(self.rated_items, self.profile)
        }
        self.non_group_items = {
            item: 0.0
            for item in self.data.tr_item - set(self.rated_items)
        }
        self.sim_non_group_members = {
            user: 0.0
            for user in set(self.data.tr_user) - set(self.rated_users)
        }

        return

    def gen_similarity(self) -> None:
        """
        计算 pseudo user 与 sim_non_group_members 中所有成员的相似度

        Args:
            None
    
        Returns：
            None

        Raises：

        """

        items_set = set(self.rated_items)

        for mem in self.sim_non_group_members:
            mem_items = self.data.tr_dict[mem]  # type:Dict[str,float]

            com_items = list(
                items_set & set(mem_items.keys()))  # type: List[str]
            if not com_items: continue  # 如果没有公共评价过的物品
            g_avg = sum(self.profile) / len(self.profile)  # type:float
            m_avg = self.data.tr_average[mem]  # type:float

            #  pseudo user 对每个项目的评分与平均评分之间的差
            g_diff = [self.item_score[item] - g_avg for item in com_items]
            #  user 评价过的物品的评分与平均评分之间的差
            u_diff = [mem_items[item] - m_avg for item in com_items]

            num1 = sum(x * y for x, y in zip(g_diff, u_diff))
            num2 = (sum(x**2 for x in g_diff) * sum(y**2 for y in u_diff))**0.5

            if num2 == 0: sim = 0.00
            else: sim = float(Decimal(num1 / num2).quantize(Decimal("0.00")))

            self.sim_non_group_members[mem] = sim

        return

    def gen_local_average(self, item: str, T: float = 0.2) -> float:
        """
        使用曼哈顿距离计算公式，计算应当使用哪些项目来避免肥尾问题,
        返回这些项目的平均评分

        Args:
            item: 物品 ID 号
            T: 最低相关度
    
        Returns：
            average:float

        Raises：
 
        """
        rate_min, rate_max = 0, 5
        diff = rate_max - rate_min
        level1, level2 = diff / 3, 2 * diff / 3

        average, count = 0.0, 0
        for g_item in self.rated_items:
            com_users = self.data.get_com_users(g_item, item)  #type:List[str]

            if not com_users: continue

            sim = 0.0
            for user in com_users:
                g_item_score = self.data.tr_dict[user][g_item]
                u_item_score = self.data.tr_dict[user][item]
                s_diff = abs(g_item_score - u_item_score)

                if s_diff < level1: sim += 1
                if level1 <= s_diff < level2: sim += 0.5
            sim /= len(com_users)

            if sim >= T:
                average += self.item_score[g_item]
                count += 1

        if count != 0:
            average = float(Decimal(average / count).quantize(Decimal("0.00")))
        return average

    def gen_recommendations(self,
                            users: List[str],
                            data: Callable,
                            k: int = 100,
                            T: float = 0.2) -> Tuple[str, float]:
        """
        为群体生成推荐

        Args:
            k：int, 推荐 k 个物品
            T: float,相似度最低下限，
                当 pseudo user 和 user 相似度 大于等于 T 时，才被用于预测评分
    
        Returns：
            average:Tuple[str,float],推荐的 k 个物品，物品 ID：预测分数

        Raises：
 
        """

        self.build(users, data)
        self.gen_similarity()
        for item in self.non_group_items:
            average = self.gen_local_average(item)
            num1, num2 = 0.0, 0.0
            for user, sim in self.sim_non_group_members.items():

                # 只计算对 item 有过评分的用户
                if item not in self.data.tr_dict[user]: continue
                # 只计算用户相似度大于 T 的用户
                if sim < T: continue
                # user 对 item 的评分
                u_item_score = self.data.tr_dict[user][item]
                # user 的平均评分
                user_avg = self.data.tr_average[user]

                num1 += sim * (u_item_score - user_avg)
                num2 += abs(sim)
            if num2 != 0: self.non_group_items[item] = average + num1 / num2

        return sorted(
            self.non_group_items.items(), key=lambda x: x[1], reverse=True)[:k]


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
    recom = MCSRecommend()
    res = recom.gen_recommendations(['67', '3', '5', '23', '276'], data)
    pprint(res)
