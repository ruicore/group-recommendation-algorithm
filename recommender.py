# @Author:             何睿
# @Create Date:        2019-03-10 10:10:49
# @Last Modified by:   何睿
# @Last Modified time: 2023-03-24 14:26:37

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Self, Tuple

from dataset import Data
from group import GroupProfile


@dataclass
class Recommend:
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

    users: list[str] = field(default_factory=list)
    data: Data = field(default_factory=Data)
    lm_profile: list[float] = field(default_factory=list)
    avg_profile: list[float] = field(default_factory=list)
    am_profile: list[float] = field(default_factory=list)
    mcs_profile: list[float] = field(default_factory=list)
    lm_score: dict[str, float] = field(default_factory=dict)
    avg_score: dict[str, float] = field(default_factory=dict)
    am_score: dict[str, float] = field(default_factory=dict)
    mcs_score: dict[str, float] = field(default_factory=dict)
    rated_items: list[str] = field(default_factory=list)
    rated_users: list[str] = field(default_factory=list)
    ng_items: dict[str, float] = field(default_factory=dict)
    sng_members: dict[str, float] = field(default_factory=dict)
    com_items: dict[str, list[str]] = field(default_factory=dict)

    def _build(self, users: list[str], data: Data) -> Self:
        """
        为对象填充数据
        获得群体抽象为个体的特征
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

        # 使用 the least misery strategy
        self.lm_score = {
            item: score
            for item, score in zip(
                self.rated_items,
                self.lm_profile,
            )
        }
        # 使用 average strategy
        self.avg_score = {
            item: score
            for item, score in zip(
                self.rated_items,
                self.avg_profile,
            )
        }
        # 使用 average without misery
        self.am_score = {
            item: score
            for item, score in zip(
                self.rated_items,
                self.am_profile,
            )
        }
        # 使用 member contribution score
        self.mcs_score = {
            item: score
            for item, score in zip(
                self.rated_items,
                self.mcs_profile,
            )
        }
        self.ng_items = {
            item: 0.0 for item in self.data.tr_item - set(self.rated_items)
        }
        self.sng_members: dict[str, float] = {
            user: 0.0 for user in set(self.data.tr_user) - set(self.rated_users)
        }

        # 计算群体评价过的物品与所有非群体成员评价过的物品的交集
        # 群体评价过的所有物品
        items_set = set(self.rated_items)

        for mem in self.sng_members:
            mem_items: dict[str, float] = self.data.tr_dict[mem]
            # 两个用户评价过的公共物品
            self.com_items[mem] = list(items_set & set(mem_items.keys()))

        return self

    def _calculate_similarity(
        self,
        profile: list[float],
        profile_dict: dict[str, float],
    ) -> Self:
        """计算 pseudo user 与 sng_members 中所有成员的相似度"""

        g_avg: float = sum(profile) / len(profile)

        if not 0 <= g_avg <= 5:
            raise Exception("g_avg is not in [0,5]")

        for member in self.sng_members:
            mem_items: dict[str, float] = self.data.tr_dict[member]
            commons = list(self.com_items[member])
            if not commons:
                self.sng_members[member] = 0
                continue  # 如果没有公共评价过的物品

            m_avg: float = self.data.tr_average[member]

            #  pseudo user 对每个项目的评分与平均评分之间的差
            g_diff = [profile_dict[item] - g_avg for item in commons]
            #  user 评价过的物品的评分与平均评分之间的差
            u_diff = [mem_items[item] - m_avg for item in commons]

            num1 = sum(x * y for x, y in zip(g_diff, u_diff))
            a, b = sum(x**2 for x in g_diff), sum(y**2 for y in u_diff)
            a, b = a**0.5, b**0.5
            num2 = a * b

            if num2 == 0:
                sim = 0
            else:
                sim = float(Decimal(num1 / num2).quantize(Decimal("0.00")))

            self.sng_members[member] = sim

        return self

    def _mla(self, item: str, scores: dict[str, float]) -> float:
        """返回用户评价过的项目的平均评分"""

        T = 0.6  # T: 最低相关度, 默认设置为 0.6
        rate_min, rate_max = 0, 5
        diff = rate_max - rate_min
        level1, level2 = diff / 3, 2 * diff / 3
        average, count = 0.0, 0

        for g_item in self.rated_items:
            # 评价过两个物品的共同用户
            coms = self.data.get_com_users(g_item, item)
            if not coms:
                continue

            sim = 0.0
            for user in coms:
                g_score = self.data.tr_dict[user][g_item]
                u_score = self.data.tr_dict[user][item]

                s_diff = abs(g_score - u_score)

                if s_diff < level1:
                    sim += 1
                elif level1 <= s_diff < level2:
                    sim += 0.5

            sim = float(Decimal(sim / len(coms)).quantize(Decimal("0.00")))

            assert sim <= 1

            if sim > T:
                average += scores[g_item]
                count += 1

        if average == 0:
            average = sum(scores.values()) / len(scores)
        else:
            average = average / count

        return float(Decimal(average).quantize(Decimal("0.00")))

    def _recommend(
        self,
        profile: List[float],
        scores: Dict[str, float],
        k: int = 500,
        num: int = 1000,
    ) -> list[tuple[str, float]]:
        """为群体生成推荐"""

        self._calculate_similarity(profile, scores)

        predict = {}
        avg = sum(profile) / len(profile)

        # 计算前 k 个最相似的用户
        neighbors: list[tuple[str, float]] = sorted(
            self.sng_members.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        for user, similarity in neighbors:
            user_avg = self.data.tr_average[user]

            for item, rate in self.data.tr_dict[user].items():
                # 只预测没有被 群体评价过分的用户
                if item not in self.ng_items:
                    continue

                predict.setdefault(item, [0, 0])
                predict[item][0] += similarity * (rate - user_avg)
                predict[item][1] += abs(similarity)

        avg = float(Decimal(avg).quantize(Decimal("0.00")))
        calculated_predict = {}
        for item in predict:
            calculated_predict[item] = self._calculate_average(predict, avg, item)

            if predict[item][1] == 0:
                calculated_predict[item] = avg
            else:
                a, b = predict[item][0], predict[item][1]
                calculated_predict[item] = float(
                    Decimal(avg + a / b).quantize(Decimal("0.00"))
                )

        return sorted(
            calculated_predict.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:num]

    def _recommend_mla(
        self,
        profile: List[float],
        scores: Dict[str, float],
        k: int = 500,
        num: int = 1000,
    ) -> list[tuple[str, float]]:
        """为群体生成推荐"""

        self._calculate_similarity(profile, scores)

        predict = {}
        # 计算前 k 个最相似的用户
        neighbors = sorted(
            self.sng_members.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        for user, sim in neighbors:
            user_avg = self.data.tr_average[user]

            for item, rate in self.data.tr_dict[user].items():
                # 只预测没有被 群体评价过分的用户
                if item not in self.ng_items:
                    continue

                predict.setdefault(item, [0, 0])
                predict[item][0] += sim * (rate - user_avg)
                predict[item][1] += sim

        calculated_predict = {}
        for item in predict:
            avg = float(Decimal(self._mla(item, scores)).quantize(Decimal("0.00")))
            calculated_predict[item] = self._calculate_average(predict, avg, item)

        return sorted(
            calculated_predict.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:num]

    @staticmethod
    def _calculate_average(
        predict: dict[str, list[float]],
        avg: float | int,
        item: str,
    ) -> float:
        if predict[item][1] == 0:
            return avg
        a, b = predict[item][0], predict[item][1]
        return float(Decimal(avg + a / b).quantize(Decimal("0.00")))

    def recommend(
        self,
        users: list[str],
        data: Data,
        k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """为群体生成推荐"""

        self._build(users, data)

        return {
            "LM": self._recommend(self.lm_profile, self.lm_score),
            "AVG": self._recommend(self.avg_profile, self.avg_score),
            "AM": self._recommend(self.am_profile, self.am_score),
            "MCS": self._recommend(self.mcs_profile, self.mcs_score),
            "MCS_MLA": self._recommend_mla(self.mcs_profile, self.mcs_score),
        }
