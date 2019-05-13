# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:10:49
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 16:44:57

import os
import csv
import math
import time
import numpy  # type: ignore
import random
import codecs
import collections
from pprint import pprint
from decimal import Decimal
from itertools import combinations
from typing import List, Dict, Set, Tuple, Callable, Generator
from dataset import Data
from group import GroupProfile


class Recommend(object):
    """
    使用基于成员贡献分数的群体推荐方式

    Attributes:
        users: List, 一个群体的所有用户
        data: 引用, 对已经对象化的数据对象的引用
        lm_profile: List, 使用 least misery strategy 生成群体 profile
        avg_profile: List, 使用 average strategy 生成群体 profile
        am_profile: List, 使用 average without misery 生成群体 profile
        mcs_profile: List, 使用 member contribution score 生成群体 profile
        lm_item_score: Dict,lm 群体抽象后对物品的评分，键为物品 ID，值为评分
        avg_item_score: Dict,avg 群体抽象后对物品的评分，键为物品 ID，值为评分
        am_item_score: Dict,am 群体抽象后对物品的评分，键为物品 ID，值为评分
        mcs_item_score: Dict,mcs 群体抽象后对物品的评分，键为物品 ID，值为评分
        rated_items: List, 群体中评过分的物品集合
        rated_users: List, users 中至少对一项物品有过评价的所有成员
        ng_items: Dict,
            non_group_items
            没有被此群体评过分的物品，即候选推荐物品集合,键为物品，值为预测评分
        sng_members: Dict,
            sim_non_group_members
            不在此群体中的其他成员,键为成员，值为该成员与 pseudo user 的相似度
    """

    def __init__(self):
        """
        建立对象

        Args:
            users: 群体成员数组
            data: 数据对象的引用
    
        """
        self.users = list()  #type:List[str]
        self.data = None  # type:Callable
        self.lm_profile = list()  # type:List[float]
        self.avg_profile = list()  # type:List[float]
        self.am_profile = list()  # type:List[float]
        self.mcs_profile = list()  # type:List[float]
        self.lm_item_score = dict()  # type:Dict[str,List[float]]
        self.avg_item_score = dict()  # type:Dict[str,List[float]]
        self.am_item_score = dict()  # type:Dict[str,List[float]]
        self.mcs_item_score = dict()  # type:Dict[str,List[float]]
        self.rated_items = list()  # List: Set[str]
        self.rated_users = list()  # List: Set[str]
        self.mla_average = dict()  # type: Dict[str,float]
        self.ng_items = dict()  # type: Dict[str,float]
        self.sng_members = dict()  # type: Dict[str,float]
        self.com_items = dict()

    def __build(self, users: List[str], data: Callable) -> None:
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
        self.lm_profile = group.lm_profile
        self.avg_profile = group.avg_profile
        self.am_profile = group.am_profile
        self.mcs_profile = group.mcs_profile
        self.rated_items = group.item_list
        self.rated_users = group.user_list

        assert len(self.avg_profile) == len(self.rated_items)

        # 使用 least misery strategy
        self.lm_item_score = {
            item: score
            for item, score in zip(self.rated_items, self.lm_profile)
        }
        # 使用 average strategy
        self.avg_item_score = {
            item: score
            for item, score in zip(self.rated_items, self.avg_profile)
        }
        # 使用 average without misery
        self.am_item_score = {
            item: score
            for item, score in zip(self.rated_items, self.am_profile)
        }
        # 使用 member contribution score
        self.mcs_item_score = {
            item: score
            for item, score in zip(self.rated_items, self.mcs_profile)
        }
        self.ng_items = {
            item: 0.0
            for item in self.data.tr_item - set(self.rated_items)
        }
        self.sng_members = {
            user: 0.0
            for user in set(self.data.tr_user) - set(self.rated_users)
        }

        del group

        # 计算群体评价过的物品与所有非群体成员评价过的物品的交集
        # 群体评价过的所有物品
        items_set = set(self.rated_items)

        for mem in self.sng_members:
            mem_items = self.data.tr_dict[mem]  # type:Dict[str,float]
            # 两个用户评价过的公共物品
            coms = list(items_set & set(mem_items.keys()))  # type: List[str]
            self.com_items[mem] = coms

        del items_set

        return

    def __gen_sim(self, profile: List[float],
                  item_score: Dict[str, float]) -> None:
        """
        计算 pseudo user 与 sng_members 中所有成员的相似度

        Args:
            profile: 群体特征, 即群体对物品的评分,list 格式，有序
            item_score: 群体特征, 即群体对物品的评分, 字典，无序
        Returns：
            None

        Raises：

        """

        # items_set = set(self.rated_items)

        g_avg = sum(profile) / len(profile)  # type:float

        for mem in self.sng_members:

            mem_items = self.data.tr_dict[mem]  # type:Dict[str,float]
            coms = self.com_items[mem]
            if not coms: continue  # 如果没有公共评价过的物品

            m_avg = self.data.tr_average[mem]  # type:float

            #  pseudo user 对每个项目的评分与平均评分之间的差
            g_diff = [item_score[item] - g_avg for item in coms]
            #  user 评价过的物品的评分与平均评分之间的差
            u_diff = [mem_items[item] - m_avg for item in coms]

            num1 = sum(x * y for x, y in zip(g_diff, u_diff))
            num2 = (sum(x**2 for x in g_diff) * sum(y**2 for y in u_diff))**0.5

            if num2 == 0: sim = -1
            else: sim = float(Decimal(num1 / num2).quantize(Decimal("0.00")))

            self.sng_members[mem] = sim

        return

    def __mla(self, item: str, scores: Dict[str, float]) -> float:
        """
        返回用户评价过的项目的平均评分

        Args:
            item: 物品 ID 号
            scores: 群体特征, 即群体对物品的评分, 字典，无序 
    
        Returns：
            average:float

        Raises：
 
        """

        T = 0.2  #  T: 最低相关度, 默认设置为 0.2
        rate_min, rate_max = 0, 5
        diff = rate_max - rate_min
        level1, level2 = diff / 3, 2 * diff / 3

        average, count = 0.0, 0
        for g_item in self.rated_items:
            # 评价过两个物品的共同用户
            coms = self.data.get_com_users(g_item, item)  #type:List[str]
            if not coms: continue

            sim = 0.0
            for user in coms:
                g_score = self.data.tr_dict[user][g_item]
                u_score = self.data.tr_dict[user][item]

                s_diff = abs(g_score - u_score)
                if s_diff < level1: sim += 1
                if level1 <= s_diff < level2: sim += 0.5
            sim /= len(coms)

            assert sim <= 1
            if sim >= T:
                average += scores[g_item]
                count += 1

        if count != 0:
            average = float(Decimal(average / count).quantize(Decimal("0.00")))
        return average

    def __recoms(
            self,
            profile: List[float],
            item_score: Dict[str, float],
            avg: float,
            k: int = 100,
            T: float = 0.2,
    ) -> Tuple[str, float]:
        """
        为群体生成推荐

        Args:
            profile: 群体特征, 即群体对物品的评分,list 格式，有序
            item_score: 群体特征, 即群体对物品的评分, 字典，无序
            k：推荐 k 个物品
            T: 相似度最低下限，
                当 pseudo user 和 user 相似度 大于等于 T 时，才被用于预测评分
            avg: 当前用户的平均评分
        Returns：
            recoms :Tuple[str,float]
                推荐的 k 个物品，物品 ID : 预测分数

        Raises：
 
        """

        self.__clear()
        self.__gen_sim(profile, item_score)
        average = avg
        for item in self.ng_items:
            num1, num2 = 0.0, 0.0
            for user, sim in self.sng_members.items():

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
            if num2 != 0: self.ng_items[item] = average + num1 / num2
        # 取前 k 个
        candiates = self.ng_items.items()
        recoms = sorted(candiates, key=lambda x: x[1], reverse=True)[:k]

        return recoms

    def __recoms_mla(
            self,
            profile: List[float],
            item_score: Dict[str, float],
            k: int = 100,
            T: float = 0.2,
    ) -> Tuple[str, float]:
        """
        为群体生成推荐

        Args:
            profile: 群体特征, 即群体对物品的评分,list 格式，有序
            item_score: 群体特征, 即群体对物品的评分, 字典，无序
            k：推荐 k 个物品
            T: 相似度最低下限，
                当 pseudo user 和 user 相似度 大于等于 T 时，才被用于预测评分
        Returns：
            recoms :Tuple[str,float]
                推荐的 k 个物品，物品 ID : 预测分数

        Raises：
 
        """

        self.__clear()
        self.__gen_sim(profile, item_score)
        for item in self.ng_items:
            average = self.__mla(item, item_score)
            num1, num2 = 0.0, 0.0
            for user, sim in self.sng_members.items():

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
            if num2 != 0: self.ng_items[item] = average + num1 / num2
        # 取前 k 个
        candiates = self.ng_items.items()
        recoms = sorted(candiates, key=lambda x: x[1], reverse=True)[:k]

        return recoms

    def __clear(self) -> None:
        """
        重置 ng_items 的值为 0.0
        由于 4 种推荐方式公用 ng_items 表，每次使用前必须清零

        Args:
            None
        Returns：
            None

        Raises：
        """

        for item, _ in self.ng_items.items():
            self.ng_items[item] = 0.0
        for usr, _ in self.sng_members.items():
            self.sng_members[usr] = 0.0
        return

    def gen_recoms(self,
                   users: List[str],
                   data: Callable,
                   k: int = 100,
                   T: float = 0.2) -> Tuple[str, float]:
        """
        为群体生成推荐

        Args:
            data: 引用, 对已经对象化的数据的引用
            users: 一个群体的所有用户
            k： 推荐 k 个物品
            T:  相似度最低下限，
                当 pseudo user 和 user 相似度 大于等于 T 时，才被用于预测评分
    
        Returns：
            recoms: Dict[str,List[Tuple[str,float]]]

        Raises：
        """
        self.__build(users, data)
        res = {}

        avg = sum(self.lm_profile) / len(self.lm_profile)
        res["LM"] = self.__recoms(self.lm_profile, self.lm_item_score, avg)

        avg = sum(self.avg_profile) / len(self.avg_profile)
        res["AVG"] = self.__recoms(self.avg_profile, self.avg_item_score, avg)

        avg = sum(self.am_profile) / len(self.am_profile)
        res["AM"] = self.__recoms(self.am_profile, self.am_item_score, avg)

        avg = sum(self.mcs_profile) / len(self.mcs_profile)
        res["MCS"] = self.__recoms(self.mcs_profile, self.mcs_item_score, avg)

        res["MCS_MLA"] = self.__recoms_mla(self.mcs_profile,
                                           self.mcs_item_score)

        return res