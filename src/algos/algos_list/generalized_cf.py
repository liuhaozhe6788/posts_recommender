# encoding:utf-8
import pandas as pd
import os
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from icecream import ic
import sys

import utils
import configs
from database import DataBase
from .basic import Basic
sys.dont_write_bytecode = True


class GeneralizedCF(object):

    def __init__(self, database, n_like_recent=380, train_data=None, test_flag=False, filter_flag=True):
        self.usr = None
        self.database = database
        self.train_data = train_data
        self.test_flag = test_flag
        self.filter_flag = filter_flag
        self.basic = Basic(self.database)
        self.n_like_recent = n_like_recent

    def choose_contents(self, set_attr: str, behavior: str, obj_attr: str, contents_from: set) -> set:
        """
        从与用户X相关的集合中获取内容，得到集合
        :param set_attr:遍历集合的主体，包括user和club
        :param behavior: 遍历集合的主体的行为
        :param obj_attr: 遍历集合的客体，包括item和selected_item
        :param contents_from: 需要遍历的集合
        :param data_base: 类数据库类的实例
        :return: 得到的内容集合
        """
        yi = set()

        # 遍历集合contents_from
        for i in contents_from:
            yi = yi | set(self.database.get_objs([set_attr, i, behavior, obj_attr], key="动态"))

        return yi

    @staticmethod
    def join_contents_set(a: set, n_a: int, b: set, n_b: int, threshold: int) -> [set, int]:
        """
        将集合a的元素部分或全部加入到b集，得到c集
        :param a: a集
        :param n_a: a集的元素个数
        :param b: b集,代表alpha或beta集
        :param n_b: b集的元素个数
        :param threshold: c集的元素个数阈值
        :return: c集和元素个数n_c
        """
        if n_a + n_b <= threshold:
            c = b | a
        else:
            a_ = set(random.sample(list(a), k=threshold - n_b))
            c = b | a_

        n_c = len(c)

        return [c, n_c]

    @staticmethod
    def sort_items(user_vec: np.ndarray, item_inv_mapper: dict, x: np.ndarray, k: int, metric='cosine',
                   show_distance=False) -> list:
        """
        使用余弦相似度对召回集进行排序
        :param user_vec: 用户特征向量
        :param item_inv_mapper: 物品反映射表
        :param x: 召回集物品的所有特征向量形成的2d矩阵
        :param k: 近邻的个数
        :param metric: 比较的量度
        :param show_distance: 是否显示与近邻的距离
        :return: 经过knn排序后的召回集
        """

        neighbour_ids = []

        knn = NearestNeighbors(n_neighbors=k, metric=metric)
        knn.fit(x)
        if isinstance(user_vec, np.ndarray):
            user_vec = user_vec.reshape(1, -1)
        neighbour = knn.kneighbors(user_vec, return_distance=show_distance)
        for i in range(0, k):
            n = neighbour.item(i)
            neighbour_ids.append(item_inv_mapper[n])
        return neighbour_ids

    def prefilter(self):
        """
        初筛过滤算法
        :return: alpha（召回集）和元素个数n_alpha
        """
        alpha = set()  # 初始化集合alpha为空
        y = set(self.database.get_objs(['user', self.usr, 'follow', 'user']))  # 关注用户集合
        # clubs_joined = set(list(map(lambda x: ":".join(x.split(":")[:-1]), data_base.get_objs(['user', user_id, 'join', 'club'], key="动态"))))
        clubs_joined = self.database.get_objs(['user', self.usr, 'join', 'club'], key="动态")

        thresholds = [50, 90, 100]

        # 取每个用户点赞的内容
        if self.test_flag:
            c_yi = set()
            for i in y:
                if i in self.train_data.keys():
                    c_yi = c_yi | set(self.train_data[i])
        else:
            c_yi = set(self.choose_contents('user', 'like', 'item', y))

        alpha = alpha | c_yi
        n_alpha = len(alpha)

        if n_alpha < thresholds[0]:
            # 取每个用户点击的内容
            d_yi = set(self.choose_contents('user', 'view', 'item', y))
            n_d_yi = len(d_yi)
            alpha, n_alpha = self.join_contents_set(d_yi, n_d_yi, alpha, n_alpha, thresholds[0])
        else:
            alpha = set(random.sample(list(alpha), k=thresholds[0]))
            n_alpha = len(alpha)

        # 取集合CLUB的所有内容精选
        e_club = set(self.choose_contents('club', 'include', 'item', clubs_joined))
        n_e_club = len(e_club)
        [alpha, n_alpha] = self.join_contents_set(e_club, n_e_club, alpha, n_alpha, thresholds[1])

        # 取集合CLUB的补集的所有内容精选
        e_club_ = set(self.choose_contents('club', 'include', 'item', set(self.database.clubs) - set(clubs_joined)))
        n_e_club_ = len(e_club_)
        [alpha, _] = self.join_contents_set(e_club_, n_e_club_, alpha, n_alpha, thresholds[2])

        # 去掉用户x已点赞，点击或曝光的内容
        if self.test_flag:
            alpha -= set(self.train_data[self.usr])
        else:
            alpha -= set(self.database.get_objs(['user', self.usr, 'like', 'item'], key="动态"))
        alpha -= set(self.database.get_objs(['user', self.usr, 'view', 'item'], key="动态"))
        alpha -= set(self.database.get_objs(['user', self.usr, 'create', 'item'], key="动态"))

        if self.filter_flag:
            alpha = self.basic.filter_club(alpha, "剧本杀")

        n_alpha = len(alpha)

        if n_alpha > thresholds[2]:
            raise ValueError(f"输出召回集alpha的长度错误，大于{thresholds[2]}")
        # print("召回集alpha的长度为：{}".format(n_alpha))
        return alpha, n_alpha

    def machine_sorting(self, alpha: set, n_alpha: int):
        """
        机器排序算法
        :param alpha: 初筛过滤得到的召回集
        :param n_alpha: 召回集alpha的元素个数
        :return: 对alpha排序得到的beta（推荐集）
        """
        if not alpha:
            return [], 0

        # 选取用户最近点赞的不多于100条内容得到用户的特征向量
        if self.test_flag:
            user_like = self.train_data[str(self.usr)]
        else:
            user_like = self.database.get_objs(['user', self.usr, 'like', 'item'], key="动态")
        n_user_like = len(user_like)
        n_recently_like = self.n_like_recent
        threshold = 100
        n_recently_like = min(n_recently_like, n_user_like)
        # clubs_list = list(dict.fromkeys(list(map(lambda x: x.split(":")[-2], list(filter(lambda s: s.startswith("动态"), data_base_.clubs))))))
        clubs_list = self.database.clubs
        n_clubs = len(clubs_list)
        if n_recently_like > 0:
            recently_like = user_like[-n_recently_like:]
            user_data = np.zeros([n_recently_like, n_clubs], dtype=float)
            for i in range(n_recently_like):
                # clubs_for_item = list(map(lambda x: x.split(":")[-2],
                #                           data_base_.get_objs(['item', recently_like[i], 'have', 'club'], key="动态")))
                clubs_for_item = self.database.get_objs(['item', recently_like[i], 'have', 'club'], key="动态")
                for club in clubs_for_item:
                    user_data[i, clubs_list.index(club)] = 1
            user_data = preprocessing.normalize(user_data, "l1")
            user_vector = user_data.sum(axis=0) / n_recently_like
        else:
            user_vector = np.array([1 / n_clubs] * n_clubs)

        # 获得物品特征向量
        items_data = np.zeros([n_alpha, n_clubs], dtype=float)
        alpha = list(alpha)
        for i in range(n_alpha):
            # print("the item is " + alpha[i])
            # clubs_for_item = list(map(lambda x: x.split(":")[-2],
            #                           data_base_.get_objs(['item', alpha[i], 'have', 'club'], key="动态")))
            clubs_for_item = self.database.get_objs(['item', alpha[i], 'have', 'club'], key="动态")
            for club in clubs_for_item:
                items_data[i, clubs_list.index(club)] = 1

        # item_mapper = dict(zip(alpha, list(range(n_alpha))))
        item_inv_mapper = dict(zip(list(range(n_alpha)), alpha))  # 将0～n_alpha-1反映射到召回集的物品id

        beta = self.sort_items(user_vector, item_inv_mapper, items_data, n_alpha, 'cosine')
        n_beta = len(beta)

        if n_beta > threshold:
            raise ValueError(f"输出推荐集beta的长度错误，大于{threshold}")
        if not n_beta == n_alpha:
            raise ValueError(f"输出推荐集beta的长度错误，与输入集alpha的长度不等")
        # print("输出推荐集beta的长度为：{}".format(n_beta))
        return beta, n_beta

    def run(self, usr):
        # print("用户ID: {}".format(self.usr))
        self.usr = usr

        alpha, n_alpha = self.prefilter()
        # print("alpha集: 元素个数{}\n{}\n".format(n_alpha, alpha))

        beta, n_beta = self.machine_sorting(alpha, n_alpha)
        # print("beta集: 元素个数{}\n{}\n".format(n_beta, beta))

        gama, n_gama = self.basic.rearrangement(beta, n_beta, self.usr, self.train_data, self.test_flag, self.filter_flag)
        # print("gama集: 元素个数{}\n{}\n".format(n_gama, gama))

        return self.database.filter("动态", gama, del_prefix=False)


if __name__ == "__main__":
    utils.create_folder_paths()
    db = DataBase(os.path.join(configs.data_folder_path, "data_20220222.xlsx"))
    generalized_cf = GeneralizedCF(db)
    generalized_cf.run("49060")
