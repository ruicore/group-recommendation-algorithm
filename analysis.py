# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:12:33
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 10:13:01

import os
import csv
import math
import time
import json
import numpy  # type: ignore
import random
import codecs
import collections
from pprint import pprint
from decimal import Decimal
from itertools import combinations
from typing import List, Dict, Set, Tuple, Callable, Generator
from dataset import Data
from recommender import Recommend


class Analysis(object):
    """
    测试类，检测使用基于成员贡献的群体推荐的效果

    Attributes:
        train: list，用于推荐的训练集
        test: list，用于测试推荐效果的测试集
    """

    def __init__(self, path: str):
        """
         Args:
            path: str,数据文件路径
        """
        self.train = list()
        self.test = list()
        self._path = path
        self._base = os.path.abspath(".")
        self.split_data(0.5)
        start = time.perf_counter()
        self.data = Data(self.train, self.test)
        end = time.perf_counter()
        print("读数据，建立对象用时{0}".format(end - start))

    def split_data(self, rate: float) -> None:
        """
        拆分数据为两个集合，一部分作为训练集，一部分作为测试集

        Args:
            rate: float,0.1-0.9，按照次比例拆分数据，rate 用于训练集，1-rate 用于测试集
    
        Returns：
            None
        
        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """
        movile_path = os.path.join(self._base, self._path)
        with codecs.open(movile_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 去掉表头
            k = rate * 100
            for row in reader:
                if random.randint(1, 101) <= k: self.train.append(row)
                else: self.test.append(row)

        return

    def __gen_group(self, g: int = 500, size: int = 5) -> Generator:
        """
        随机生成 g 个成员个数为 size 的群，返回 Generator

        Args:
            g: int, 生成测试群的个数
            size:int, 每组人数的大小
    
        Returns：
            None
        
        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """
        for _ in range(g):
            yield random.sample(self.data.tr_user, k=size)
        return

    def __gen_avg_ndcg(self, recoms_set: Set[str], recoms_dict: Dict[str, int],
                       users: List[str]) -> float:
        """
        Args:
            recoms_set: Set[str],给群体的推荐集合
            recoms_dict: Dict[str, int], 给群体的推荐集合
                键为物品，值为该物品在推荐列表中的顺序
            users: List[str],需要计算 ndcg 值的所有用户
    
        Returns：
            一个群体的 ndcg 平均值所有成员的 ndcg_score 值
        
        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """

        average = 0.00

        for user in users:
            average += self.__gen_ndcg(recoms_set, recoms_dict, user)
        if average != 0.0:
            average = Decimal(average / (len(users))).quantize(Decimal("0.00"))

        return float(average)

    def __gen_ndcg(self, recoms_set: Set[str], recoms_dict: Dict[str, int],
                   user: str) -> float:
        """
        Args:
            recoms_set: Set[str],给群体的推荐集合
            recoms_dict: Dict[str, int], 给群体的推荐集合
                键为物品，值为该物品在推荐列表中的顺序
            user: str,当前需要计算 ndcg 值的用户
    
        Returns：
            当前成员的 ndcg_score 值
        
        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """

        coms = set(self.data.te_dict[user].keys()) & recoms_set
        if not coms: return 0.0

        # 用户 u 对物品的真实评分
        u_item_score = {item: self.data.te_dict[user][item] for item in coms}
        # 以物品评分降序排列构成的列表，计算 IDCG
        s = sorted(u_item_score.items(), key=lambda x: x[1], reverse=True)
        IDCG = self._gen_dcg(s)

        coms_dict = dict()
        for item, _ in self.data.te_dict[user].items():
            if item in coms:
                # 键：物品，值：该物品在推荐列表中的索引
                coms_dict[item] = recoms_dict[item]

        # 拿到在给群体的推荐物品中，用户 u 评价过的物品
        # 这些物品在推荐列表中从前到后有序排列
        s = sorted(coms_dict.items(), key=lambda x: x[1], reverse=False)
        DCG = self._gen_dcg(s)

        return float(Decimal(DCG / IDCG).quantize(Decimal("0.00")))

    def _gen_dcg(self, item_score: List[Tuple[str, float]]) -> float:
        """
        对一个序列计算 DCG 值
        
        Args:
            item_score: List[Tuple[物品 id, 评分]]
    
        Returns：
            当前序列的 DCG 值
        
        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """

        DCG = item_score[0][1]
        for index, item in enumerate(item_score[1:]):  # index 从 0 开始
            score = item[1]
            DCG += score / math.log2(index + 2)  # index 需要从 2 开始
        return DCG

    def __gen_f(self,
                users: [str],
                recoms: List[Tuple[str, float]],
                T: float = 3.0) -> float:
        """
        计算推荐序列的 F 值
            for item_i
            TP: when all the member ratings for item_i are higer then T
                and the gropu prediction for item_i is higer than T,
                then TP is count as 1.
            FN: when all the member ratings for item_i are higer then T
                and the gropu prediction for item_i is lower than T, 
                then FN is count as 1.
            FP: when some the member ratings for item_i are lower then T
                and the gropu prediction for item_i is higer than T,
                then TP is count as 1.
            F = 2 * (TP / (TP + FN)) * (TP / (TP + FP))
        
        Args:
            users: [str],群体的所有成员
            recoms: List[Tuple[str,float]],给群体的推荐集合
            T:float,threshod to determinate whether we should accept one item or not.

        Returns：
            当前序列的 F 值
        
        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """
        TP, FN, FP = 0, 0, 0
        for item, score in recoms:
            higer, lower = 0, 0

            for u in users:
                # u 没有对 item 评价过分数
                if item not in self.data.te_dict[u]: continue
                rel_score = self.data.te_dict[u][item]
                if rel_score >= T: higer += 1
                else: lower += 1

            if higer == 0 and lower == 0:
                # print("没有评分记录")
                continue
            # 预测分大于 T 且所有成员评分大于 T
            if score >= T and lower == 0: TP += 1
            # 预测分大于 T 且存在成员对该物品的评分小于 T
            if score >= T and lower != 0: FP += 1
            # 预测分小于 T 且所有成员对该物品的评分大于 T
            if score < T and lower == 0: FN += 1

        if TP + FP == 0 or TP / (TP + FP) == 0: return 0.00
        F = 2 * (TP / (TP + FN)) * (TP / (TP + FP))
        return float(Decimal(F).quantize(Decimal("0.00")))

    def assess(self, ):
        """
        对一个序列计算 DCG 值
        
        Args:
            item_score: List[Tuple[物品 id, 评分]]
    
        Returns：
            当前序列的 DCG 值
        
        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """

        methods = ["LM", "AVG", "AM", "MCS", "MCS_MLA"]
        metrics = ["ndcg", "f"]
        min_size, max_size, step = 5, 5, 5

        rates = {key: {m: list() for m in metrics} for key in methods}

        recomend_engine = Recommend()

        # size 由  min_size 增加到 max_size,步长为 step
        for size in range(min_size, max_size + 1, step):
            # 每类生成 10 个群体
            for users in self.__gen_group(g=10, size=size):

                start = time.perf_counter()
                recoms = recomend_engine.gen_recoms(users, self.data)
                end = time.perf_counter()
                g_items = len(recomend_engine.lm_item_score)

                print("为大小为{0}的群体使用 5 种方法生成群体推荐用时{1},此群体中一共有 {2} 个项目".format(
                    size, end - start, g_items))

                for m in methods:
                    # 推荐物品集合
                    re_set = set(item[0] for item in recoms[m])
                    re_dict = {item[0]: item[1] for item in recoms[m]}

                    # 推荐物品字典
                    ndcg = self.__gen_avg_ndcg(re_set, re_dict, users)
                    rates[m][metrics[0]].append(ndcg)
                    f = self.__gen_f(users, recoms[m])
                    rates[m][metrics[1]].append(f)

        path = os.path.join(self._base, "rates.json")
        with codecs.open(path, "a") as file:
            file.write(json.dumps(rates))


if __name__ == "__main__":
    analysis = Analysis(r"movies\movies_small\ratings.csv")
    analysis.assess()
