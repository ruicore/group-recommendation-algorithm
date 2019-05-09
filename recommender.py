# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:10:49
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 16:44:57

import time
import math
import random
import itertools


class Data(object):
    """
    准备数据集合

    Attributes:
        train: dict,所有类都将使用的训练数据
        test: dict,所有类都将使用的测试数据
    """

    def __init__(self):
        movies = Movies()
        self.train, self.test = movies.spilt_data(M=8, k=7, seed=1)


class UserSimilarity(Data):
    def __init__(self, ):
        super().__init__()

    def _consine_sim(self) -> dict:
        """
        基于用户的协同过滤算法
        使用余弦相似度计算用户之间的相似度

        Args:
            train: dict, {user1:{item1:value,item2:value ...}...}

        Returns:
            W:dict, {user1:{user2:value,user2:value...}...} 用户之间的相似度大小
        """

        # 建立 item -- users 关系表
        # item_users：{item1:{user1,user2...}...}
        item_users = dict()
        for user, items in self.train.items():
            for i in items.keys():
                if i not in item_users: item_users[i] = set()
                item_users[i].add(user)
        # C: dict, C[u][v] 表示用户 u 和用户 v 购买过的商品交集的个数
        # N: dict, N[u] 表示用户 u 购买过的商品总数
        C, N = dict(), dict()
        for users in item_users.values():
            for u in users:
                # 统计用户 u 购买过的商品的总数
                N[u] = N.get(u, 0) + 1
                for v in users:
                    if u == v: continue
                    if u not in C: C[u] = dict()
                    C[u][v] = C[u].get(v, 0) + 1
        # W：dict，用户 u 和 v 之间的余弦相似度字典，可以看作为一个二维矩阵
        W = dict()
        for u, related_users in C.items():
            for v, cuv in related_users.items():
                if u not in W: W[u] = dict()
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return W

    def _consine_sim_ii(self) -> dict:
        """
        基于用户的协同过滤算法
        使用改进的余弦相似度计算用户之间的相似度
        
        Args:
            train: dict, {user1:{item1:value,item2:value ...}...}
        Returns:
            W:dict, {user1:{user2:value,user2:value...}...} 用户之间的相似度大小
        """

        item_users = dict()
        for user, items in self.train.items():
            for i in items.keys():
                if i not in item_users: item_users[i] = set()
                item_users[i].add(user)
        # C:dict, C[u]     [v] 表示用户 u 和用户 v 购买过的商品交集的个数
        # N:dict, N[u] 表示用户 u 购买过的商品总数
        C, N = dict(), dict()
        for users in item_users.values():
            for u in users:
                N[u] = N.get(u, 0) + 1
                for v in users:
                    if u == v: continue
                    if u not in C: C[u] = dict()
                    C[u][v] = C[u].get(v, 0) + 1 / math.log(1 + len(users))
        W = dict()
        for u, related_users in C.items():
            for v, cuv in related_users.items():
                if u not in W: W[u] = dict()
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return W


class ItemSimilarity(Data):
    def __init__(self):
        super().__init__()

    def _consine_sim(self):
        """
        基于物品的协同过滤算法
        使用余弦相似度计算物品之间的相似度

        Args:
            train: dict, {user1:{item1:value,item2:value ...}...}
            C: dict, C[i][j] 表示购买了 i 物品也购买了 j 物品额用户个数
            N: dict, N[i] 表示物品 i 被用户购买够的总次数

        Returns:
            W:dict, {user1:{user2:value,user2:value...}...} 用户之间的相似度大小
        """

        C, N = dict(), dict()
        for items in self.train.values():
            for i in items:
                # 统计产品 i 被购买过的总次数
                N[i] = N.get(i, 0) + 1
                for j in items:
                    if i == j: continue
                    if i not in C: C[i] = dict()
                    # 产品 i 和产品 j 被用户同时购买的次数
                    C[i][j] = C[i].get(j, 0) + 1
        W = dict()
        for i, related_items in C.items():
            for j, cij in related_items.items():
                if i not in W: W[i] = dict()
                W[i][j] = cij / math.sqrt(N[i] * N[j])
        return W

    def _consine_sim_ii(self) -> dict:
        """
        基于物品的协同过滤算法
        使用改进的余弦相似度计算物品之间的相似度

        Args:
            train: dict, {user1:{item1:value,item2:value ...}...}
            C: dict, C[i][j] 表示购买了 i 物品也购买了 j 物品额用户个数
            N: dict, N[i] 表示物品 i 被用户购买够的总次数
        
        Returns:
            W:dict, {user1:{user2:value,user2:value...}...} 用户之间的相似度大小
        """

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
                if i not in W: W[i] = dict()
                W[i][j] = cij / math.sqrt(N[i] * N[j])
        return W


class LatentFactorModel(Data):
    """
    隐语义模型计算相似度
    """

    def __init__(self):
        pass

    def _random_select_negative_sample(self, items: dict) -> dict:
        """
        随机负样本采样

        Args:
            itmems: dict

        Returns:
            ret: dict
        """
        ret = {i: 1 for i in items.keys()}
        n = 0
        items_pool = []
        for _ in range(0, len(items) * 3):
            item = items_pool[random.randint(0, len(items_pool) - 1)]
            if item in ret: continue
            ret[item] = 0
            n += 1
            if n > len(items): break
        return ret



class Recommend(Data):
    def __init__(self):
        super().__init__()
        self.user_w = UserSimilarity()._consine_sim()
        self.item_w = ItemSimilarity()._consine_sim()

    def user_cos_recommend(self, user: str, K=10) -> dict:
        """
        基于用户协同过滤的算法，为单个用户 user 提供推荐

        Args:
            user: str, 用户名称
            K: int, 取前 K 个最相似的用户，利用这些用户提供推荐

        Returns:
            rank:dict,{item1:sim,item2:sim} 返回建议的物品和该物品的推荐值（相似度）
        """

        rank = dict()
        # rank: {item:score}
        interacted_items = self.train.get(user, dict())
        # 取和用户 user 相似度最高的前 K 个用户
        for v, wuv in sorted(
                self.user_w[user].items(), key=lambda x: x[1],
                reverse=True)[0:K]:
            for i, rvi in self.train[v].items():
                if i in interacted_items: continue
                rank[i] = rank.get(i, 0) + wuv * rvi
        # rank 中推荐的项目个数与 K 无关
        return rank

    def item_cos_recommend(self, user: str, K=10) -> dict:
        """
        基于物品的协同过滤算法，为用户 user 提供推荐

        Args:
            user: str,用户名称
            K: int, 取前 K 个最相似的物品，利用这些物品提供推荐

        Returns:
            rank:dict,{item1:sim,item2:sim} 返回建议的物品和该物品的推荐值（相似度）
        """

        rank = dict()
        ru = self.train.get(user, dict())
        for i, pi in ru.items():
            for j, wj in sorted(
                    self.item_w[i].items(), key=lambda x: x[1],
                    reverse=True)[0:K]:
                if j in ru: continue
                rank[j] = rank.get(j, 0) + pi * wj
        return rank


if __name__ == "__main__":
    recom = Recommend()
    print(recom.user_cos_recommend("2"))