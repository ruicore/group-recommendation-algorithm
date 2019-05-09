# -*- coding: utf-8 -*-
# @Author:             何睿
# @Create Date:        2019-03-10 10:12:33
# @Last Modified by:   何睿
# @Last Modified time: 2019-03-10 10:13:01

import os
import csv
import math
import codecs
import random
from recommend import Recommend


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
        self.train = []
        self.test = []
        self._path = path
        self._base = os.path.abspath(".")
        self.split_data(0.5)

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


if __name__ == "__main__":
    analysis = Analysis(r"movies_small\ratings.csv")
    print(analysis.train)