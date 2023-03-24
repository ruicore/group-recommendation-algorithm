# @Author:             何睿
# @Create Date:        2019-03-10 10:12:33
# @Last Modified by:   何睿
# @Last Modified time: 2023-03-24 14:26:37

import codecs
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Generator, List, Set, Tuple

from dataset import Data
from recommender import Recommend


@dataclass
class Analysis:
    """测试类，检测使用基于成员贡献的群体推荐的效果"""

    path: str
    train: list[list[str]] = field(default_factory=list)
    test: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.base = os.path.abspath(".")
        self.split_data(0.5)
        start = time.perf_counter()
        self.data = Data(self.train, self.test)
        end = time.perf_counter()
        print("读数据，建立对象用时 {:10}".format(end - start))

    def split_data(self, rate: float) -> None:
        """拆分数据为两个集合，一部分作为训练集，一部分作为测试集"""
        movie_path = os.path.join(self.base, self.path)
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

    def _generate_group(
        self,
        g: int = 500,
        size: int = 5,
    ) -> Generator[list[str], None, None]:
        """随机生成 g 个成员个数为 size 的群，返回 Generator"""
        for _ in range(g):
            yield random.sample(self.data.te_user, k=size)

    def _calculate_avg_ndcg(
        self,
        recommendation_set: set[str],
        recommendation_dict: dict[str, int],
        users: list[str],
    ) -> float:
        """
        recommendation_set: 给群体的推荐集合
        recommendation_dict: 给群体的推荐集合，键为物品，值为该物品在推荐列表中的顺序
        users: 需要计算 ndcg 值的所有用户
        """

        average, count = 0.00, 0

        for user in users:
            ndcg = self._calculate_ndcg(recommendation_set, recommendation_dict, user)
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

    def _calculate_ndcg(
        self,
        recommendation_set: Set[str],
        recommendation_dict: Dict[str, int],
        user: str,
    ) -> float:
        """
        recommendation_set: 给群体的推荐集合
        recommendation_dict: 给群体的推荐集合，键为物品，值为该物品在推荐列表中的顺序
        user: str,当前需要计算 ndcg 值的用户
        """

        coms = set(self.data.te_dict[user].keys()) & recommendation_set

        # coms 为空说明给 user 推荐的物品全部出现在了训练集中，无法计算这个用户的 ndcg
        if not coms:
            return -1

        # 计算理想 DCG 值
        # 用户 u 对物品的真实评分
        u_item_score = {item: self.data.te_dict[user][item] for item in coms}

        # 以物品评分降序排列构成的列表，计算 IDCG
        s = sorted(u_item_score.items(), key=lambda x: x[1], reverse=True)
        IDCG = self._calculate_dcg(s)

        # 计算真实 DCG 值
        # {物品：物品在推荐序列中的序号} 排序用
        coms_dict = {item: recommendation_dict[item] for item in coms}

        # 获取 user 评价过的物品在推荐序列中的顺序
        s = [
            (item[0], self.data.te_dict[user][item[0]])
            for item in sorted(coms_dict.items(), key=lambda x: x[1], reverse=False)
        ]

        DCG = self._calculate_dcg(s)

        assert IDCG >= DCG

        return float(Decimal(DCG / IDCG).quantize(Decimal("0.00")))

    @staticmethod
    def _calculate_dcg(item_score: list[tuple[str, float]]) -> float:
        """对一个序列计算 DCG 值"""

        DCG = 0.00
        for index, item in enumerate(item_score):  # index 从 0 开始
            DCG += (2 ** item[1] - 1) / math.log2(index + 2)  # index 需要从 2 开始
        return DCG

    def _calculate_f(
        self,
        users: list[str],
        recommendation_info: list[tuple[str, float]],
        T: float = 3.7,
    ) -> float:
        """
        计算推荐序列的 F 值
            for item_i
            TP: when all the member ratings for item_i are higher then T
                and the group prediction for item_i is higher than T,
                then TP is count as 1.
            FN: when all the member ratings for item_i are higher then T
                and the group prediction for item_i is lower than T,
                then FN is count as 1.
            FP: when some the member ratings for item_i are lower than T
                and the group prediction for item_i is higher than T,
                then TP is count as 1.
            F = 2 * (TP / (TP + FN)) * (TP / (TP + FP))

        """
        TP, FN, FP, TN = 0, 0, 0, 0

        for item, score in recommendation_info:
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

    def report(
        self,
        g: int = 1000,
        min_size: int = 5,
        max_size: int = 30,
        step: int = 5,
    ) -> None:
        """评价不同推荐算法的性能"""

        methods = ["LM", "AVG", "AM", "MCS", "MCS_MLA"]
        metrics = ["nDCG", "F"]
        recommend_engine = Recommend()

        avg_rates = {key: {method: list() for method in methods} for key in metrics}

        # size 由  min_size 增加到 max_size,步长为 step
        for size in range(min_size, max_size + 1, step):
            rates = {key: {m: list() for m in metrics} for key in methods}
            # 每类生成 g 个群体
            count = 0
            for users in self._generate_group(g=g, size=size):
                count += 1

                start = time.perf_counter()
                recommend = recommend_engine.recommend(users, self.data)
                end = time.perf_counter()

                g_items = len(recommend_engine.lm_score)
                print(
                    "群体大小： {:2} ,第{:4}  个群体, 项目数： {:4}, 推荐用时： {:8}".format(
                        size, count, g_items, end - start
                    )
                )

                for m in methods:
                    # 推荐物品集合
                    re_set = {item[0] for item in recommend[m]}
                    # 推荐集合，被推荐物品 : 物品在推荐序列的索引
                    re_dict = {
                        item[0]: index for index, item in enumerate(recommend[m])
                    }

                    ndcg = self._calculate_avg_ndcg(re_set, re_dict, users)
                    # 特殊标记，ndcg 为 -1 说明所有推荐的物品在测试集合中没有出现过一次
                    # 无法对此次推荐做评价
                    if ndcg != -1:
                        rates[m][metrics[0]].append(ndcg)

                    f = self._calculate_f(users, recommend[m])
                    # 特殊标记，f 为 -1 说明所有推荐的物品在测试集合中没有出现过一次
                    # 无法对此次推荐做评价
                    if f != -1:
                        rates[m][metrics[1]].append(f)

            for m in methods:
                if len(rates[m][metrics[0]]):
                    avg = sum(rates[m][metrics[0]]) / len(rates[m][metrics[0]])
                    print("{:5}{:8}{:10.3}".format("nDCG", m, avg))
                    avg_rates["nDCG"][m].append(avg)

            for m in methods:
                if len(rates[m][metrics[1]]):
                    avg = sum(rates[m][metrics[1]]) / len(rates[m][metrics[1]])
                    print("{:5}{:8}{:10.3}".format("F", m, avg))
                    avg_rates["F"][m].append(avg)

            path = os.path.join(self.base, "rates" + str(size) + ".json")
            with codecs.open(path, "w") as file:
                file.write(json.dumps(rates))

        path = os.path.join(self.base, "avg_rates.json")
        with codecs.open(path, "w", encoding="utf-8") as file:
            file.write(json.dumps(avg_rates))


if __name__ == "__main__":
    analysis = Analysis(r"movies/ratings.csv")  # 从 movie lens 下载的 ratings.csv 数据集
    analysis.report(g=10, min_size=5, max_size=30)
