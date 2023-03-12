# encoding:utf-8
import os
import random
import numpy as np
import numpy.ma as ma
import pandas as pd
import feather
from sklearn.metrics.pairwise import pairwise_distances
from icecream import ic
import sys
sys.dont_write_bytecode = True

import configs
import utils
from database import DataBase
from .basic import Basic
np.set_printoptions(threshold=np.inf)


class ItemCF():

    def __init__(self, database, buffer_name, sim_thres=0, pred_thres=0, train_data=None, test_flag=False, filter_flag=True):
        self.usr = None
        self.database = database  # 存储所有数据的数据库类
        self.train_data = train_data  # 用于性能指标计算的训练数据
        self.test_flag = test_flag  # 模型测试的flag
        self.filter_flag = filter_flag  # 过滤保留“剧本杀”动态的flag
        self.like_users = self.database.like_users  # 所有点赞用户
        self.liked_dynamics = self.database.liked_dynamics  # 所有被点赞的动态
        self.like_users_map = self.map_list_to_index(self.like_users)   # 点赞用户编号到行索引的映射
        self.liked_dynamics_map = self.map_list_to_index(self.liked_dynamics)   # 被点赞的动态编号到列索引的映射
        self.like_users_reverse_map = self.map_index_to_list(self.like_users)   # 点赞用户行索引到用户编号的映射
        self.liked_dynamics_reverse_map = self.map_index_to_list(self.liked_dynamics)  # 被点赞的动态列索引到动态编号的映射
        self.basic = Basic(self.database)   # basic类用于调用一些基本操作的成员函数
        self.buffer_name = buffer_name  # buffer文件名，buffer文件存储所有用户对物品的预测值
        self.sim_thres = sim_thres  # 相似度阈值用来选择近邻
        self.pred_thres = pred_thres    # 预测阈值用来选择top-n推荐的个数

    @staticmethod
    def map_list_to_index(a: list) -> dict:
        """
        列表元素映射为枚举的字典
        """
        return dict(zip(tuple(a), range(len(a))))

    @staticmethod
    def map_index_to_list(a: list) -> dict:
        """
        枚举映射为列表元素的字典
        """
        return dict(zip(range(len(a)), tuple(a)))

    def create_ui_matrix(self) -> np.ndarray:
        """
        创建用户对物品的点赞矩阵
        """
        ui_matrix = np.zeros((len(self.like_users), len(self.liked_dynamics)), dtype="float32")
        for u in self.like_users:
            if self.test_flag:
                like_items = self.train_data[str(u)]
            else:
                like_items = self.database.get_objs(["user", u, 'like', 'item'], key="动态")
            # like_items_map = [dynamics_map[s] for s in like_items]
            for i in like_items:
                ui_matrix[self.like_users_map[u]][self.liked_dynamics_map[i]] = 1
            # like_items_index = ui_matrix[users_map[u]][:].nonzero()
        return ui_matrix

    def compare_and_filter(self, sim):
        """
        物品的相似度矩阵中低于self.sim_thres的元素置为0
        """
        thres_matrix = np.ones(sim.shape) * self.sim_thres
        mask_matrix = np.greater(sim, thres_matrix)
        return np.where(mask_matrix, sim, 0)

    def calc_item_similarities(self, ui_matrix: np.ndarray) -> np.ndarray:
        """
        计算物品的相似度矩阵
        """
        item_similarities_ = pairwise_distances(ui_matrix.T, metric="cosine")
        item_similarities_ = np.ones(item_similarities_.shape) - item_similarities_
        item_similarities_filtered_ = self.compare_and_filter(item_similarities_)

        return item_similarities_filtered_

    @staticmethod
    def predict(ui_matrix: np.ndarray, item_similarities_: np.ndarray) -> np.ndarray:
        """
        预测用户对未知物品的评分
        """
        pred = ui_matrix.dot(item_similarities_) / np.array([np.abs(item_similarities_).sum(axis=0)])
        return pred

    def get_pred(self):
        ui_matrix = self.create_ui_matrix()
        item_similarities_ = self.calc_item_similarities(ui_matrix)
        pred = self.predict(ui_matrix, item_similarities_)
        return pred

    def get_top_n(self):
        """
        实现基于物品的协同过滤算法，并将所有用户的前N名推荐结果存入.feather文档
        :return:None
        """

        def _get_item_list(row, reverse_map):
            num_items = len(self.liked_dynamics)
            if len(row) == num_items * 2 + 1:
                mindex = ma.masked_less(np.array(row[num_items: -1]), self.pred_thres)
                ma.set_fill_value(mindex, 0)
                item_indices = np.nonzero(mindex)[0]
                # item_indices = list(item_indices - np.ones(len(item_indices)) * num_items)
                items = [reverse_map[i] for i in list(row[item_indices])]
                if self.test_flag:
                    like_items = self.train_data[str(row["user"])]
                else:
                    like_items = self.database.get_objs(["user", str(row["user"]), 'like', 'item'], key="动态")
                view_items = self.database.get_objs(['user', str(row["user"]), 'view', 'item'], key="动态")
                create_items = self.database.get_objs(['user', str(row["user"]), 'create', 'item'], key="动态")
                items = [i for i in items if (i not in like_items) and (i not in view_items) and (i not in create_items)]
                return ",".join(items)
            else:
                raise ValueError(f"top_n_df的列数应该为{num_items * 2 + 1}，实际为{len(row)}")
        pred = self.get_pred()
        sorted_item_pred = -np.sort(-pred)  # 将pred进行降序排序
        sorted_item_index = np.argsort(-pred)  # 根据pred，对每个用户的物品进行降序排序
        top_n_df = pd.DataFrame(data=sorted_item_index)
        pred_df = pd.DataFrame(data=sorted_item_pred)
        cols = pred_df.columns
        pred_df = pred_df.rename(dict(zip(list(cols), list(range(len(cols), 2*len(cols))))), axis='columns')
        top_n_df = pd.concat([top_n_df, pred_df], axis=1)
        top_n_df["user"] = np.arange(len(top_n_df))
        top_n_df["user"] = top_n_df["user"].apply(lambda x: self.like_users_reverse_map[x])  # 根据行索引增加用户栏
        top_n_df["top_n"] = top_n_df.apply(lambda x: _get_item_list(x, self.liked_dynamics_reverse_map),
                                           axis=1)  # 获得每位用户的top_n物品列表

        top_n_df = top_n_df[["user", "top_n"]]
        top_n_df = top_n_df.astype({"user": str})
        if self.buffer_name.endswith(".feather"):
            feather.write_dataframe(top_n_df, os.path.join(configs.buffer_files_folder_path,
                                                           self.buffer_name))
        elif self.buffer_name.endswith(".xlsx"):
            top_n_df.to_excel(os.path.join(configs.buffer_files_folder_path, self.buffer_name))
        else:
            raise ValueError("buffer文件后缀名错误")
        return None

    def run(self, usr):
        """
        对user_id用户进行物品推荐，该过程读取.feather文档，找到用户对应的前N名物品随后增加最受欢迎的物品
        :param usr:
        :return:
        """
        self.usr = usr
        if self.buffer_name.endswith(".feather"):
            top_n_df = feather.read_dataframe(os.path.join(configs.buffer_files_folder_path,
                                                           self.buffer_name))
        elif self.buffer_name.endswith(".xlsx"):
            top_n_df = pd.read_excel(os.path.join(configs.buffer_files_folder_path, self.buffer_name))
        else:
            raise ValueError("buffer文件后缀名错误")
        top_n_df = top_n_df.astype({"user": str})
        top_n = top_n_df.loc[(top_n_df["user"] == str(self.usr)),"top_n"].values
        if not top_n:
            top_n_items = []
        elif top_n[0] == '' or pd.isna(top_n[0]):
            top_n_items = []
        else:
            top_n_items = top_n[0].split(",")

        if self.filter_flag:
            top_n_items = self.basic.filter_club(top_n_items, "剧本杀")

        gama, n_gama = self.basic.rearrangement(top_n_items, len(top_n_items), self.usr, self.train_data, self.test_flag, self.filter_flag)

        return self.database.filter("动态", gama, del_prefix=False)


