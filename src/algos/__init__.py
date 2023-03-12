# encoding:utf-8
"""
    该package用于对所有用户或者单个用户运行选中的算法，和计算所有算法的性能指标
"""
from .algos_list import GeneralizedCF, ItemCF, UserCF, HybridCF, Basic
from .algos_operator import AlgosOperator
