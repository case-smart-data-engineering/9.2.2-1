import numpy as np
import pandas as pd
from pgmpy.estimators import (BicScore, HillClimbSearch,
                              MaximumLikelihoodEstimator)
from pgmpy.models import BayesianNetwork

# 创建一个BN的类
class BN():
    def __init__(self, V=[], E=[]):
        # BN节点集合
        self.V = V
        # BN边集合
        self.E = E


# 输出各节点的条件概率
def mle(V, E, D):
    V = list(V)
    E = list(E)
    data = pd.DataFrame(np.array(D), columns=V)
    model = BayesianNetwork(E)
    for v in V:
        # 最大似然估计
        cpd_v = MaximumLikelihoodEstimator(model, data).estimate_cpd(v)
        # 输出各节点的条件概率
        print(cpd_v)

# BN结构学习
def BNsearch(g, D):
    V = g.V
    E = g.E
    data = pd.DataFrame(np.array(D), columns=V)

    # 使用爬山法学习BN结构
    est = HillClimbSearch(data)
    best_model = est.estimate(scoring_method=BicScore(data))
    E = best_model.edges()
    g.E = E

    # 输出各节点的条件概率
    mle(g.V, g.E, D)

    return g


if __name__ == '__main__':
    # 数据节点集合
    V = ['A', 'B', 'C']
    # 数据集
    D = [['F', 'F', 'F'], ['T', 'T', 'T'], \
            ['T', 'T', 'F'], ['F', 'F', 'F']]
    g = BN(V=V)
    # 学习BN结构
    G = BNsearch(g, D)
    # 输出BN的边
    print(G.E)
