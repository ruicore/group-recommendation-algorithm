# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:09:59
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 10:16:40

# https://blog.csdn.net/lanchunhui ，m/article/details/49494265
from dataset import Data


class GroupProfile(Data):
    """
        Generate Group file
    """

    def __init__(self):
        super().__init__()
        self.profile = {}
        self.data = {
            "user1": [5, 4, 4, 0],
            "user2": [4, 4, 0, 4],
            "user3": [0, 0, 5, 2],
            "user4": [3, 1, 0, 3]
        }

    def get_profile(self):
        """
        计算群体的特征
        """
        weight = {}
        for matrix in self.get_combinations():
            for user in self.get_represent():
                weight[user] = weight.get(user) + 1
        _sum = sum(weight.values())
        weight = {user: weight[user] / _sum for user in weight}
        for user in weight:
            weight[user] = weight[user] / _sum

    def get_combinations(self):
        """
            根据一个二维矩阵生成列的排列组合
        """
        return None

    def get_represent(self, matrix) -> dict():
        """
        根据二维矩阵，判断哪些用户是代表用户
        """
        return None
