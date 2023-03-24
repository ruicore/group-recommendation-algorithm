# @Author:             何睿
# @Create Date:        2019-03-10 10:11:29
# @Last Modified by:   何睿
# @Last Modified time: 2022-05-05 15:37:53

from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations
from typing import Dict, List, Set


@dataclass
class Data:
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

    tr_data: list[list[str]] = field(default_factory=list)
    te_data: list[list[str]] = field(default_factory=list)
    tr_dict: dict[str, dict[str, float]] = field(default_factory=dict)
    te_dict: dict[str, dict[str, float]] = field(default_factory=dict)
    tr_user: list[str] = field(default_factory=list)
    tr_item: set[str] = field(default_factory=set)
    te_user: list[str] = field(default_factory=list)
    te_item: set[str] = field(default_factory=set)
    tr_average: dict[str, float] = field(default_factory=dict)
    tr_item_com_users: dict[str, dict[str, set[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """建立对象"""

        self._build(self.tr_data, self.tr_dict, self.tr_user, self.tr_item)
        self._build(self.te_data, self.te_dict, self.te_user, self.te_item)
        self._build_item_common_users()
        self._build_average()

    @classmethod
    def _build(
        cls,
        data: List[List[str]],
        table: Dict[str, Dict[str, float]],
        user_list: List[str],
        item_set: Set[str],
    ) -> None:
        """
        构建 用户-项目 评分表
        构建所有的用户表
        构建所有的物品表
        """

        uid, mid, rid = range(3)  # 用户 id ，电影 id，评分 id 分别对应的索引
        for line in data:
            user, item, rating = line[uid], line[mid], float(line[rid])
            if user not in table:
                table[user] = dict()
                user_list.append(user)

            table[user][item] = rating
            item_set.add(item)

        return

    def _build_item_common_users(self) -> None:
        """构建 tr_item_com_users 表,此表用于存储评价过两个项目的公共用户索引"""

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

    def _build_average(self) -> None:
        """计算用户对评价过的所有物品的评分平均数"""

        for user, items in self.tr_dict.items():
            sum_score, count = sum(items.values()), len(items)
            num = Decimal(sum_score / count).quantize(Decimal("0.00"))
            self.tr_average[user] = float(num)
        return

    def get_com_users(self, item1: str, item2: str) -> List[str]:
        """返回评价过两个物品的用户交集"""
        items = sorted([item1, item2])
        it1, it2 = items[0], items[1]
        commons = []
        if it1 in self.tr_item_com_users and it2 in self.tr_item_com_users[it1]:
            commons = list(self.tr_item_com_users[it1][it2])
        return commons
