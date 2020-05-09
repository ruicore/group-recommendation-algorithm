# Group Recommender System

![pyton3](https://img.shields.io/badge/language-python3-blue.svg) &nbsp;[![codebeat badge](https://codebeat.co/badges/234570ae-9f92-400b-85db-54b8df6bdc2d)](https://codebeat.co/projects/github-com-ruicore-group-recommendation-algorithm-master) &nbsp; ![issue](https://img.shields.io/github/issues/ruicore/group-recommendation-algorithm) &nbsp; ![forks](https://img.shields.io/github/forks/ruicore/group-recommendation-algorithm) &nbsp; ![stars](https://img.shields.io/github/stars/ruicore/group-recommendation-algorithm) &nbsp; ![license](https://img.shields.io/github/license/ruicore/group-recommendation-algorithm) &nbsp; ![twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2Fruicore%2Fgroup-recommendation-algorithm)

基于成员贡献的群体推荐系统

## Table of Contents

- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)


## Usage

* 本论文将群体转换成为个体，然后再利用基于用户的协同过滤算法进行推荐。
* 利用余弦相似度来确定群体成员的贡献程度，确定其贡献分数，最后生成群体评分向量。
* 使用曼哈顿距离计算公式来计算局部平均分，用来代替全局平均分，用于缓解推荐系统中存在的长尾分布问题。
* 使用 NDCG 和 F 指标来评测推荐算法的好坏。

## Maintainers

[@RuiCore](https://github.com/RuiCore)

## Contributing

PRs accepted.

Small note: If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## License

LGPL-3.0 © 2019 RuiCore
