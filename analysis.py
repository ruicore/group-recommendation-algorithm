# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:12:33
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 10:13:01

import math
from recommend import Recommend


class Test(Recommend):
    """
    测试类，检验训练得到的结果
    """

    def __init__(self):
        super().__init__()
        # 为用户推荐的 item 个数
        self.n = 10

    def recall(self) -> float:
        """
        计算召回率

        Args:
            None
    
        Returns：
            float,召回率
        """

        hit, _all = 0, 0
        for user in self.train.keys():
            tu = self.test.get(user, dict())
            for item in self.user_cos_recommend(user):
                if item in tu:
                    hit += 1
            _all += len(tu)
        return hit / _all

    def precision(self) -> float:
        """
        计算准确率

        Args:
            None
    
        Returns：
            float,准确率
        """

        hit, _all = 0, 0
        for user in self.train.keys():
            tu = self.test.get(user, dict())
            recommend_items = self.user_cos_recommend(user)
            for item in recommend_items:
                if item in tu: hit += 1
            _all += len(recommend_items)
        return hit / _all

    def coverage(self) -> float:
        """
        计算覆盖率

        Args:
            None
    
        Returns：
            float,覆盖率
        """

        recommend_items = set()
        all_items = set()
        for user in self.train.keys():
            for item in self.train[user].keys():
                all_items.add(item)
            for item in self.user_cos_recommend(user):
                recommend_items.add(item)
        return len(recommend_items) / len(all_items)

    def popularity(self) -> float:
        """
        计算流行度

        Args:
            None
    
        Returns：
            float，流行度
        """

        item_popularity = dict()
        for items in self.train.values():
            for item in items.keys():
                item_popularity[item] = item_popularity.get(item, 0) + 1
        ret, n = 0, 0
        for user in self.train.keys():
            rank = self.user_cos_recommend(user)
            for item in rank:
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / n


if __name__ == "__main__":
    test = Test()
    recall = test.recall()
    precision = test.precision()
    coverage = test.coverage()
    popularity = test.popularity()
    print("准确率:", precision, "召回率：", recall, "覆盖率:", coverage, "流行度",
          popularity)
