# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:12:33
# @Last Modified by:   何睿
# @Last Modified time: 2022-05-05 15:37:53

import codecs
import csv
import json
import math
import os
import random
import time
from decimal import Decimal
from typing import Dict, Generator, List, Set, Tuple

import numpy  # type: ignore

from dataset import Data
from recommender import Recommend


class Analysis:
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
        print("读数据，建立对象用时 {0:10}".format(end - start))

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
        movie_path = os.path.join(self._base, self._path)
        with codecs.open(movie_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # 去掉表头
            k = rate * 100
            for row in reader:
                if random.randint(1, 101) <= k:
                    self.train.append(row)
                else:
                    self.test.append(row)

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
            yield random.sample(self.data.te_user, k=size)
        return

    def __gen_avg_ndcg(
            self,
            recoms_set: Set[str],
            recoms_dict: Dict[str, int],
            users: List[str],
    ) -> float:
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

        average, count = 0.00, 0

        for user in users:
            ndcg = self.__gen_ndcg(recoms_set, recoms_dict, user)
            # ndcg == -1 说明无法计算此用户的 ndcg 值
            if ndcg == -1:
                continue

            average += ndcg
            count += 1

        # count 为 0 说明此群体中每个人的 ndcg 值都无法计算，因此无法评价对此群体推荐的好坏
        if count == 0:
            return -1

        average = Decimal(average / count).quantize(Decimal("0.00"))

        return float(average)

    def __gen_ndcg(
            self,
            recoms_set: Set[str],
            recoms_dict: Dict[str, int],
            user: str,
    ) -> float:
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

        # coms 为空说明给 user 推荐的物品全部出现在了训练集中，无法计算这个用户的 ndcg
        if not coms:
            return -1

        # 计算理想 DCG 值
        # 用户 u 对物品的真实评分
        u_item_score = {item: self.data.te_dict[user][item] for item in coms}

        # 以物品评分降序排列构成的列表，计算 IDCG
        s = sorted(u_item_score.items(), key=lambda x: x[1], reverse=True)
        IDCG = self.__gen_dcg(s)

        # 计算真实 DCG 值
        # {物品：物品在推荐序列中的序号} 排序用
        coms_dict = {item: recoms_dict[item] for item in coms}

        # 获取 user 评价过的物品在推荐序列中的顺序
        s = [
            (item[0], self.data.te_dict[user][item[0]])
            for item in sorted(coms_dict.items(), key=lambda x: x[1], reverse=False)
        ]

        DCG = self.__gen_dcg(s)

        assert IDCG >= DCG

        return float(Decimal(DCG / IDCG).quantize(Decimal("0.00")))

    def __gen_dcg(self, item_score: List[Tuple[str, float]]) -> float:
        """
        对一个序列计算 DCG 值

        Args:
            item_score: List[Tuple[物品 id, 评分]]

        Returns：
            当前序列的 DCG 值

        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """

        DCG = 0.00
        for index, item in enumerate(item_score):  # index 从 0 开始
            DCG += (2 ** item[1] - 1) / math.log2(index + 2)  # index 需要从 2 开始
        return DCG

    def __gen_f(
            self,
            users: [str],
            recoms: List[Tuple[str, float]],
            T: float = 3.7,
    ) -> float:
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
        TP, FN, FP, TN = 0, 0, 0, 0

        for item, score in recoms:
            higer, lower = 0, 0

            for u in users:
                # u 没有对 item 评价过分数
                if item not in self.data.te_dict[u]:
                    continue

                rel_score = self.data.te_dict[u][item]
                if rel_score >= T:
                    higer += 1
                else:
                    lower += 1

            if higer == 0 and lower == 0:
                continue

            # 预测分大于 T 且所有成员评分大于 T
            if score >= T and lower == 0:
                TP += 1
            # 预测分大于 T 且存在成员对该物品的评分小于 T
            if score >= T and lower != 0:
                FP += 1
            # 预测分小于 T 且所有成员对该物品的评分大于 T
            if score < T and lower == 0:
                FN += 1

            if score < T and higer == 0:
                TN += 1

        # 如果所有的值都为 0，说明所有推荐的物品在测试集合中没有出现过一次
        # 无法对此次推荐做评价

        if TP == 0 and FP == 0 and FN == 0 and TN == 0:
            return -1

        if TP == 0:
            return 0.00
        precision, recall = TP / (TP + FP), TP / (TP + FN)
        F = 2 * precision * recall / (precision + recall)

        return float(Decimal(F).quantize(Decimal("0.00")))

    def assess(
            self,
            g: int = 1000,
            min_size: int = 5,
            max_size: int = 30,
            step: int = 5,
    ) -> None:
        """
        评价不同推荐算法的性能

        Args:
            g : 每类群生成的数量
            min_size : 起始群体成员数量
            max_size : 最大群体成员数量
            step : 每次递增

        Returns：
            当前序列的 DCG 值

        Raises：
            IOError: An error occurred accessing the bigtable.Table object.
        """

        methods = ["LM", "AVG", "AM", "MCS", "MCS_MLA"]
        metrics = ["nDCG", "F"]
        recomend_engine = Recommend()

        avg_rates = {key: {method: list() for method in methods} for key in metrics}

        # size 由  min_size 增加到 max_size,步长为 step
        for size in range(min_size, max_size + 1, step):
            rates = {key: {m: list() for m in metrics} for key in methods}
            # 每类生成 g 个群体
            count = 0
            for users in self.__gen_group(g=g, size=size):

                count += 1

                start = time.perf_counter()
                recoms = recomend_engine.recoms(users, self.data)
                end = time.perf_counter()

                g_items = len(recomend_engine.lm_score)
                print(
                    "群体大小： {0:2} ,第{1:4}  个群体, 项目数： {2:4}, 推荐用时： {3:8}".format(
                        size, count, g_items, end - start
                    )
                )

                for m in methods:
                    # 推荐物品集合
                    re_set = set(item[0] for item in recoms[m])
                    # 推荐集合，被推荐物品 : 物品在推荐序列的索引
                    re_dict = {item[0]: index for index, item in enumerate(recoms[m])}

                    ndcg = self.__gen_avg_ndcg(re_set, re_dict, users)
                    # 特殊标记，ndcg 为 -1 说明所有推荐的物品在测试集合中没有出现过一次
                    # 无法对此次推荐做评价
                    if ndcg != -1:
                        rates[m][metrics[0]].append(ndcg)

                    f = self.__gen_f(users, recoms[m])
                    # 特殊标记，f 为 -1 说明所有推荐的物品在测试集合中没有出现过一次
                    # 无法对此次推荐做评价
                    if f != -1:
                        rates[m][metrics[1]].append(f)

            for m in methods:
                if len(rates[m][metrics[0]]):
                    avg = sum(rates[m][metrics[0]]) / len(rates[m][metrics[0]])
                    print("{0:5}{1:8}{2:10.3}".format("nDCG", m, avg))
                    avg_rates["nDCG"][m].append(avg)

            for m in methods:
                if len(rates[m][metrics[1]]):
                    avg = sum(rates[m][metrics[1]]) / len(rates[m][metrics[1]])
                    print("{0:5}{1:8}{2:10.3}".format("F", m, avg))
                    avg_rates["F"][m].append(avg)

            path = os.path.join(self._base, "rates" + str(size) + ".json")
            with codecs.open(path, "w") as file:
                file.write(json.dumps(rates))

        path = os.path.join(self._base, "avg_rates.json")
        with codecs.open(path, "w", encoding="utf-8") as file:
            file.write(json.dumps(avg_rates))


if __name__ == "__main__":
    analysis = Analysis(r"movies\ratings.csv")  # 从 movie lens 下载的 ratings.csv 数据集
    analysis.assess(g=10, min_size=5, max_size=30)
