# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:10:49
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 16:44:57

import time
import math
import random
import itertools
from dataset import Movies


class Data(object):
    """
    准备数据集合
    train:dict,{user1:{item1:value,item2:value ...}...}
    """

    def __init__(self):
        movies = Movies()
        self.train, self.test = movies.spilt_data(M=8, k=7, seed=1)


class UserSimilarity(Data):
    def __init__(self, ):
        super().__init__()

    def _consine_sim(self) -> dict:
        """
        基于用户的协同过滤
        余弦相似度计算用户之间的相似度
        trian:{user:{item:count}}
        """
        item_users = dict()
        for user, items in self.train.items():
            for i in items.keys():
                if i not in item_users: item_users[i] = set()
                item_users[i].add(user)
        # C:dict, C[u][v] 表示用户 u 和用户 v 购买过的商品交集的个数
        # N:dict,N[u] 表示用户 u 购买过的商品总数
        C, N = dict(), dict()
        for users in item_users.values():
            for u in users:
                N[u] = N.get(u, 0) + 1
                for v in users:
                    if u == v: continue
                    if u not in C: C[u] = dict()
                    C[u][v] = C[u].get(v, 0) + 1
        # W：dict，用户 u 和 v 之间的余弦相似度
        W = dict()
        for u, related_users in C.items():
            for v, cuv in related_users.items():
                if u not in W: W[u] = dict()
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return W

    def _consine_sim_ii(self) -> dict:
        """
        基于用户的协同过滤
        基于余弦相似度，惩罚用户 u 和用户 v 共同兴趣列表中
        的热门物品
        """
        item_users = dict()
        for user, items in self.train.items():
            for i in items.keys():
                if i not in item_users: item_users[i] = set()
                item_users[i].add(user)
        # C:dict, C[u][v] 表示用户 u 和用户 v 购买过的商品交集的个数
        # N:dict,N[u] 表示用户 u 购买过的商品总数
        C, N = dict(), dict()
        for i, users in item_users.items():
            for u in users:
                N[u] = N.get(u, 0) + 1
            for v in users:
                if u == v: continue
                if u not in C: C[u] = dict()
                C[u][v] = C[u].get(v, 0) + 1 / math.log(1 + len(users))
        W = dict()
        for user, related_users in C.items():
            for v, cuv in related_users.items():
                if user not in W: W[user] = dict()
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return W


class ItemSimilarity(Data):
    def __init__(self):
        super().__init__()

    def _consine_sim(self):
        # C:dici,C[i][j] 表示购买了 i 物品也购买了 j 物品额用户个数
        # N[i] 表示物品 i 被用户购买够的总次数
        C, N = dict(), dict()
        for items in self.train.values():
            for i in items:
                N[i] = N.get(i, 0) + 1
                for j in items:
                    if i == j: continue
                    if i not in C: C[i] = dict()
                    C[i][j] = C[i].get(j, 0) + 1
        W = dict()
        for i, related_items in C.items():
            for j, cij in related_items():
                if i not in W: W[i] = dict()
                W[i][j] = cij / math.sqrt(N[i] * N[j])
        return W

    def _consine_sim_ii(self) -> dict:
        C, N = dict(), dict()
        for items in self.train.values():
            for i in items:
                N[i] = N.get(i, 0) + 1
                for j in items:
                    if i == j: continue
                    if i not in C: C[i] = dict()
                    C[i][j] = 1 / math.log(1 + len(items))

        W = dict()
        for i, related_items in C.items():
            for j, cij in related_items.items():
                W[i][j] = cij / math.sqrt(N[i] * N[j])
        return W


class Recommend(Data):
    def __init__(self):
        super().__init__()

    def user_cos_recommend(self, user: str) -> dict:
        """
        为用户 user 提供推荐
        """
        K = 10
        rank = dict()
        # rank: {item:score}
        W = UserSimilarity()._consine_sim()
        interacted_items = self.train.get(user, None)
        # 取和用户 user 相似度最高的前 K 个用户
        for v, wuv in sorted(
                W[user].items(), key=lambda x: x[1], reverse=True)[0:K]:
            for i, rvi in self.train[v].items():
                if i in interacted_items: continue
                rank[i] = rank.get(i, 0) + wuv * rvi
        # rank 中推荐的项目个数与 K 无关
        return rank

    def item_cos_recommend(self, user: str) -> dict:
        K = 10
        rank = dict()
        ru = self.train.get(user, dict())
        W = ItemSimilarity()._consine_sim()
        for i, pi in ru.items():
            for j, wj in sorted(
                    W[i].items(), key=lambda x: x[1], reverse=True)[0:K]:
                if j in ru: continue
                rank[j] = rank.get(j, 0) + pi * wj
        return rank


if __name__ == "__main__":
    recom = Recommend()
    print(recom.user_cos_recommend("2"))