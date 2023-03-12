# encoding:utf-8
"""
    该package用于实现所有推荐算法，包括：
    1.广义协同过滤（generalized_cf）
    2.基于用户的协同过滤（user_cf)
    3.基于物品的协同过滤（item_cf)
    4.基于用户与物品的混合协同过滤（hybrid_cf)
"""
from .generalized_cf import GeneralizedCF
from .item_cf import ItemCF
from .user_cf import UserCF
from .hybrid_cf import HybridCF
from .basic import Basic
