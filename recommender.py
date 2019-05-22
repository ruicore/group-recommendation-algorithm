# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:10:49
# @Last Modified by:   何睿
# @Last Modified time: 2019-05-14 20:05:46

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
        lm_score: Dict,lm 群体抽象后对物品的评分，键为物品 ID，值为评分
        avg_score: Dict,avg 群体抽象后对物品的评分，键为物品 ID，值为评分
        am_score: Dict,am 群体抽象后对物品的评分，键为物品 ID，值为评分
        mcs_score: Dict,mcs 群体抽象后对物品的评分，键为物品 ID，值为评分
        rated_items: List, 群体中评过分的物品集合
        rated_users: List, users 中至少对一项物品有过评价的所有成员
        ng_items: Dict,
            non_group_items
            没有被此群体评过分的物品，即候选推荐物品集合,键为物品，值为预测评分
        sng_members: Dict,
            sim_non_group_members
            不在此群体中的其他成员,键为成员，值为该成员与 pseudo user 的相似度
        com_items: Dict, 用户 g 和所有用户评价过物品的交集，键为用户，值为交集
    """

    def __init__(self)->None:
        """
        建立对象

        Args:
            users: 群体成员数组
            data: 数据对象的引用
    
        """
        self.users = list()  #type:List[str]
        self.data = None  # type:Callable
        self.lm_profile = list()  # type:List[float]
        self.avg_profile = list()  # type: List[float]
        self.am_profile = list()  # type: List[float]
        self.mcs_profile = list()  # type: List[float]
        self.lm_score = dict()  # type: Dict[str,float]
        self.avg_score = dict()  # type: Dict[str,float]
        self.am_score = dict()  # type: Dict[str,float]
        self.mcs_score = dict()  # type: Dict[str,float]
        self.rated_items = list()  # type: List[str]
        self.rated_users = list()  # type: List[str]
        self.mla_average = dict()  # type: Dict[str,float]
        self.ng_items = dict()  # type: Dict[str,float]
        self.sng_members = dict()  # type: Dict[str,float]
        self.com_items = dict()  # type: Dict[str,List[str]]

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
        self.lm_score = {
            item: score
            for item, score in zip(self.rated_items, self.lm_profile)
        }
        # 使用 average strategy
        self.avg_score = {
            item: score
            for item, score in zip(self.rated_items, self.avg_profile)
        }
        # 使用 average without misery
        self.am_score = {
            item: score
            for item, score in zip(self.rated_items, self.am_profile)
        }
        # 使用 member contribution score
        self.mcs_score = {
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
                  profile_dict: Dict[str, float]) -> None:
        """
        计算 pseudo user 与 sng_members 中所有成员的相似度

        Args:
            profile: 群体特征, 即群体对物品的评分,list 格式，有序
            profile_dict: 群体特征, 即群体对物品的评分, 字典，无序
        Returns：
            None

        Raises：

        """

        g_avg = sum(profile) / len(profile)  # type:float

        assert 0 <= g_avg <= 5
        for mem in self.sng_members:

            mem_items = self.data.tr_dict[mem]  # type:Dict[str,float]
            coms = list(self.com_items[mem])
            if not coms:
                self.sng_members[mem] = 0
                continue  # 如果没有公共评价过的物品

            m_avg = self.data.tr_average[mem]  # type:float

            #  pseudo user 对每个项目的评分与平均评分之间的差
            g_diff = [profile_dict[item] - g_avg for item in coms]
            #  user 评价过的物品的评分与平均评分之间的差
            u_diff = [mem_items[item] - m_avg for item in coms]

            num1 = sum(x * y for x, y in zip(g_diff, u_diff))
            a, b = sum(x**2 for x in g_diff), sum(y**2 for y in u_diff)
            a, b = a**0.5, b**0.5
            num2 = a * b

            if num2 == 0: sim = 0
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

        T = 0.8  #  T: 最低相关度, 默认设置为 0.8
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
                
                if s_diff < level1: 
                    sim += 1
                elif level1 <= s_diff < level2: 
                    sim += 0.5
            
            sim = float(Decimal(sim /len(coms)).quantize(Decimal("0.00")))

            assert sim <= 1

            if sim > T:
                average += scores[g_item]
                count += 1

        # assert average !=0

        if average == 0:
            average = sum(scores.values()) / len(scores)
            print("曼哈顿无效")
        else:
            average = average / count 

        return float(Decimal(average).quantize(Decimal("0.00")))

    def __recoms(self, profile: List[float], scores: Dict[str, float], k: int = 30, num: int = 100) -> List[Tuple[str, float]]:
        """
        为群体生成推荐

        Args:
            profile: 群体特征, 即群体对物品的评分,list 格式，有序
            scores: 群体特征, 即群体对物品的评分, 字典，无序
            k：使用前 k 个相似的用户  
            num: 推荐 num 个 item              
            
        Returns：
            recoms :Tuple[str,float]
                推荐的 k 个物品，物品 ID : 预测分数

        Raises：
 
        """

        self.__gen_sim(profile, scores)

        predict = dict() # type:Dict[str,float]
        avg = sum(profile) / len(profile)

        # 计算前 k 个最相似的用户
        neighbors = sorted(self.sng_members.items(), key=lambda x: x[1], reverse=True)[:k]  # type:List[Tuple[str,float]]

        for user, sim in neighbors:
            user_avg = self.data.tr_average[user]

            for item, rate in self.data.tr_dict[user].items():
                # 只预测没有被 群体评价过分的用户
                if item not in self.ng_items: continue

                predict.setdefault(item, [0, 0])
                predict[item][0] += sim * (rate - user_avg)
                predict[item][1] += sim


        avg = float(Decimal(avg).quantize(Decimal("0.00")))

        for item in predict:
            if predict[item][1] == 0:
                predict[item] = avg
            else:
                a, b = predict[item][0], predict[item][1]
                predict[item] = float(Decimal(avg + a / b).quantize(Decimal("0.00")))
            
        recoms = sorted(predict.items(), key=lambda x: x[1], reverse=True)[:num]

        return recoms

    def __recoms_mla(self, profile: List[float], scores: Dict[str, float],k: int = 30, num: int = 30) -> List[Tuple[str, float]]:
        """
        为群体生成推荐

        Args:
            profile: 群体特征, 即群体对物品的评分,list 格式，有序
            scores: 群体特征, 即群体对物品的评分, 字典，无序
            k：使用前 k 个相似的用户  
            num: 推荐 num 个 item              
            
        Returns：
            recoms :Tuple[str,float]
                推荐的 k 个物品，物品 ID : 预测分数

        Raises：
 
        """

        self.__gen_sim(profile, scores)

        predict = dict() # type:float,Dict[str,float]

        # 计算前 k 个最相似的用户
        neighbors = sorted(self.sng_members.items(), key=lambda x: x[1],reverse=True)[:k]  # type:List[Tuple[str,float]]

        for user, sim in neighbors:
            user_avg = self.data.tr_average[user]

            for item, rate in self.data.tr_dict[user].items():
                # 只预测没有被 群体评价过分的用户
                if item not in self.ng_items: continue

                predict.setdefault(item, [0, 0])
                predict[item][0] += sim * (rate - user_avg)
                predict[item][1] += sim


        for item in predict:
            avg = float(Decimal(self.__mla(item, scores)).quantize(Decimal("0.00")))
            
            if predict[item][1] == 0:
                predict[item] = avg
            else:
                a, b = predict[item][0], predict[item][1]
                predict[item] = float(Decimal(avg + a / b).quantize(Decimal("0.00")))

        recoms = sorted(predict.items(), key=lambda x: x[1], reverse=True)[:num]

        return recoms

    def recoms(self, users: List[str], data: Callable, k: int = 100) -> Dict[str,List[Tuple[str, float]]]:
        """
        为群体生成推荐

        Args:
            data: 引用, 对已经对象化的数据的引用
            users: 一个群体的所有用户
            k： 推荐 k 个物品
    
        Returns：
            recoms: Dict[str,List[Tuple[str,float]]]

        Raises：
        """

        self.__build(users, data)
        res = {}

        # print("LM")
        res["LM"] = self.__recoms(self.lm_profile, self.lm_score)
        # print("AVG")
        res["AVG"] = self.__recoms(self.avg_profile, self.avg_score)
        # print("AM")
        res["AM"] = self.__recoms(self.am_profile, self.am_score)
        # print("MCS")
        res["MCS"] = self.__recoms(self.mcs_profile, self.mcs_score)
        # print("MCS_MLA")
        res["MCS_MLA"] = self.__recoms_mla(self.mcs_profile, self.mcs_score)

        return res