# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:12:33
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 10:13:01

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
        self._path = path
        self.train = list()
        self.test = list()
        self._base = os.path.abspath(".")
        self.split_data(0.5)
        self.data = Data(self.train, self.test)

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

    def gen_group(self, g: int = 500, size: int = 5) -> Generator:
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

    def gen_ndcg(self, recoms_set: Set[str], recoms_dict: Dict[str, int],
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

    def gen_f(self,
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
            rel_score = self.data.te_dict[u][item]
            for u in users:
                if rel_score >= T: higer += 1
                else: lower += 1
            if higer == 0 and lower == 0:
                print("没有评分记录")
                continue
            # 预测分大于 T 且所有成员评分大于 T
            if score >= T and lower == 0: TP += 1
            # 预测分大于 T 且存在成员对该物品的评分小于 T
            if score >= T and lower != 0: FP += 1
            # 预测分小于 T 且所有成员对该物品的评分大于 T
            if score < T and lower == 0: FN += 1

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
        methods =["LM","AVG","AM","MCS"]
        min_size, max_size, step = 5, 30, 5
        for size in range(min_size, max_size + 1, step):
            for group in self.gen_group(size):
                pass


if __name__ == "__main__":
    analysis = Analysis(r"movies\movies_small\ratings.csv")
    analysis.assess()
