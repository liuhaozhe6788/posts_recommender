# encoding:utf-8
import pandas as pd
import numpy as np
import os
import sys
import random
import copy
import itertools
from icecream import ic

import utils
import configs
from .algos_list import GeneralizedCF, ItemCF, UserCF, HybridCF, Basic
sys.dont_write_bytecode = True

random.seed(0)


class AlgosOperator(object):
    """
        运行所有推荐算法、计算推荐算法的性能指标和调整算法模型的参数
    """

    def __init__(self, database, filter_flag=True):
        self.database = database
        self.filter_flag = filter_flag
        self.basic = Basic(self.database)
        self.user_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "user_cf_params.npy"))
        self.item_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "item_cf_params.npy"))
        self.hybrid_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "hybrid_cf_params.npy"))
        utils.create_folder_paths()

    def run_all_algos(self) -> list:
        """
        运行所有推荐算法得到推荐结果
        :return: 推荐结果列表，列表元素为一个推荐算法给所有用户的推荐结果为pd.DataFrame型
        """
        algo_res_list = []
        for algo_name in ['generalized_cf', 'item_cf', 'user_cf', 'hybrid_cf']:
            res = self.run_all_users(algo_name)
            algo_res_list.append(res)
        return algo_res_list

    def run_all_users(self, algo_name, item_cat="动态"):
        """
        对所有点赞用户做推荐，推荐结果存储在.xlsx表格中
        :param algo_name:算法名称
        :param item_cat:物品类别，默认为动态
        :return:所有用户的推荐结果
        """
        if item_cat == "动态":
            gama_sets = []

            if algo_name == "generalized_cf":
                generalized_cf = GeneralizedCF(self.database)
                for usr in self.database.like_users:
                    gama = generalized_cf.run(usr)
                    gama_sets.append(",".join(gama))

            elif algo_name == "item_cf":
                item_cf = ItemCF(self.database, "item_cf_top_n_recommendation_map.feather", self.item_cf_params[0], self.item_cf_params[1])
                item_cf.get_top_n()

                for usr in self.database.like_users:
                    gama = item_cf.run(usr)
                    gama_sets.append(",".join(gama))

            elif algo_name == "user_cf":
                user_cf = UserCF(self.database, "user_cf_top_n_recommendation_map.feather", self.user_cf_params[0], self.user_cf_params[1])
                user_cf.get_top_n()

                for usr in self.database.like_users:
                    gama = user_cf.run(usr)
                    gama_sets.append(",".join(gama))

            elif algo_name == "hybrid_cf":
                hybrid_cf = HybridCF(self.database,
                                     "item_cf_in_hybrid_cf_top_n_recommendation_map.feather",
                                     "user_cf_in_hybrid_cf_top_n_recommendation_map.feather",
                                     self.hybrid_cf_params[0]
                                     )
                hybrid_cf.get_top_n()
                for usr in self.database.like_users:
                    gama = hybrid_cf.run(usr)
                    gama_sets.append(",".join(gama))

            gama_df = pd.DataFrame(data={"user": self.database.like_users, "recommendation": gama_sets})
            gama_df.to_excel(os.path.join(configs.rec_result_folder_path, f"{algo_name}_recommendation_map.xlsx"),
                             index=False)
            gama_df["recommendation"] = gama_df.apply(lambda x: x["recommendation"].split(","), axis=1)
            return gama_df

    # 将database中的点赞行为数据划分为训练集和测试集
    def train_test_split(self, train_ratio=0.5):
        for usr in self.database.like_users:
            like_items = self.database.get_objs(['user', usr, 'like', 'item'], key="动态")
            like_indices = len(like_items)
            train_indices = sorted(random.sample(range(like_indices), k=int(train_ratio * like_indices) + 1))
            test_indices = [i for i in range(like_indices) if i not in train_indices]
            train_like_items = [like_items[i] for i in train_indices]
            test_like_items = [like_items[i] for i in test_indices]
            if self.filter_flag:
                test_like_items = self.basic.filter_club(test_like_items, "剧本杀")

            yield [usr, train_like_items, test_like_items]

    # 对数据进行2-fold分区
    def two_fold_partition(self):
        for usr in self.database.like_users:
            like_items = self.database.get_objs(['user', usr, 'like', 'item'], key="动态")
            like_indices = len(like_items)

            yield [usr, like_items[: int(like_indices/2)], like_items[int(like_indices/2):]]

    # 计算训练集与验证集的比例
    def calc_train_test_ratio(self):
        train_size = 0
        test_size = 0
        for usr, train_like_items, test_like_items in self.train_test_split(0.5):
            train_size += len(train_like_items)
            test_size += len(test_like_items)
        return train_size/test_size

    # 使用训练集的数据训练模型，得到推荐结果
    def get_recommendation(self, algo_name):
        recommend_data = []
        train_data = []
        test_data = []

        for usr, train_like_items, test_like_items in self.train_test_split(0.5):
            train_data.append((usr, train_like_items))
            for test_like_item in test_like_items:
                # if '剧本杀' in list(map(lambda x: x.split(":")[-2], data_base_.get_objs(['item', test_like_item, 'have', 'club'], key="动态"))):
                test_data.append([usr, test_like_item])
        train_data = dict(train_data)

        if algo_name == "generalized_cf":
            for usr in self.database.like_users:
                generalized_cf = GeneralizedCF(self.database, train_data=train_data, test_flag=True)
                gama = generalized_cf.run(usr)
                for recommend_item in gama:
                    recommend_data.append([usr, recommend_item])

        elif algo_name == "item_cf":
            item_cf_ = ItemCF(self.database, "item_cf_top_n_recommendation_map.feather", self.item_cf_params[0],
                              self.item_cf_params[1], train_data, True)
            item_cf_.get_top_n()

            for usr in self.database.like_users:
                gama = item_cf_.run(usr)
                for recommend_item in gama:
                    recommend_data.append([usr, recommend_item])

        elif algo_name == "user_cf":
            user_cf_ = UserCF(self.database, "user_cf_top_n_recommendation_map.feather", self.user_cf_params[0],
                              self.user_cf_params[1], train_data, True)
            user_cf_.get_top_n()

            for usr in self.database.like_users:
                gama = user_cf_.run(usr)
                for recommend_item in gama:
                    recommend_data.append([usr, recommend_item])

        elif algo_name == "hybrid_cf":
            hybrid_cf = HybridCF(self.database,
                                 "item_cf_in_hybrid_cf_top_n_recommendation_map.feather",
                                 "user_cf_in_hybrid_cf_top_n_recommendation_map.feather",
                                 self.hybrid_cf_params[0],
                                 train_data,
                                 True
                                 )
            hybrid_cf.get_top_n()

            for usr in self.database.like_users:
                gama = hybrid_cf.run(usr)
                for recommend_item in gama:
                    recommend_data.append([usr, recommend_item])

        test_df = pd.DataFrame(data=test_data)
        test_df = test_df.rename(columns={0: "user", 1: "item"})
        recommend_df = pd.DataFrame(data=recommend_data)
        recommend_df = recommend_df.rename(columns={0: "user", 1: "item"})
        return test_df, recommend_df

    # 使用验证集的数据进行推荐精度、命中率/召回率和F-score的计算
    def calc_metrics(self, algo_name):
        test_df, recommend_df = self.get_recommendation(algo_name)
        tp_df = pd.merge(test_df, recommend_df, on=["user", "item"])
        accuracy_ = len(tp_df) / len(recommend_df)
        hit_ratio_ = len(tp_df) / len(test_df)
        f_score_ = 2 * accuracy_ * hit_ratio_ / (accuracy_ + hit_ratio_)
        return [accuracy_, hit_ratio_, f_score_]

    # 计算某个推荐算法的覆盖率
    def calc_coverage(self, algo_name):
        rec_res = self.run_all_users(algo_name, item_cat="动态")
        num_all_script_kill_items = len(self.basic.filter_club(self.database.items, "剧本杀"))
        num_all_rec_items = len(list(set(itertools.chain.from_iterable(rec_res["recommendation"].to_list()))))
        return num_all_rec_items/num_all_script_kill_items

    # 调整GeneralizedCF模型的参数
    def generalizedCF_tuning(self, n_like_recent_range):
        train_parts = [[], []]
        test_parts = [[], []]
        f_scores = []

        for usr, like_items_p1, like_items_p2 in self.two_fold_partition():
            train_parts[0].append((usr, like_items_p1))
            test_like_items = copy.deepcopy(like_items_p2)
            if self.filter_flag:
                test_like_items = self.basic.filter_club(like_items_p2, "剧本杀")
            for test_like_item in test_like_items:
                test_parts[0].append([usr, test_like_item])

            train_parts[1].append((usr, like_items_p2))
            test_like_items = copy.deepcopy(like_items_p1)
            if self.filter_flag:
                test_like_items = self.basic.filter_club(like_items_p1, "剧本杀")
            for test_like_item in test_like_items:
                test_parts[1].append([usr, test_like_item])

        for i in range(len(train_parts)):
            train_parts[i] = dict(train_parts[i])

        for j in range(len(n_like_recent_range)):
            f_score_parts = []
            for k in range(len(train_parts)):
                recommend_data = []

                generalized_cf = GeneralizedCF(self.database, n_like_recent=n_like_recent_range[j],
                                               train_data=train_parts[k], test_flag=True)

                for usr in self.database.like_users:
                    gama = generalized_cf.run(usr)
                    for recommend_item in gama:
                        recommend_data.append([usr, recommend_item])

                test_df = pd.DataFrame(data=test_parts[k])
                test_df = test_df.rename(columns={0: "user", 1: "item"})
                recommend_df = pd.DataFrame(data=recommend_data)
                recommend_df = recommend_df.rename(columns={0: "user", 1: "item"})
                tp_df = pd.merge(test_df, recommend_df, on=["user", "item"])

                accuracy = len(tp_df) / len(recommend_df)
                hit_ratio = len(tp_df) / len(test_df)

                f_score = 2 * accuracy * hit_ratio / (accuracy + hit_ratio)
                f_score_parts.append(f_score)
            f_scores.append(sum(f_score_parts)/len(f_score_parts))
        f_scores = np.array(f_scores)

        result = np.where(f_scores == np.amax(f_scores))
        param_index = result[0][0]
        param = [n_like_recent_range[param_index]]
        np.save(os.path.join(configs.tuning_result_folder_path, "generalized_cf_tuning.npy"), f_scores)
        np.save(os.path.join(configs.tuning_result_folder_path, "generalized_cf_params.npy"), np.asarray(param))
        return param, f_scores

    # 调整userCF模型的参数
    def userCF_tuning(self, num_neigh_thres_range, pred_thres_range):
        train_parts = [[], []]
        test_parts = [[], []]
        f_scores = []

        for usr, like_items_p1, like_items_p2 in self.two_fold_partition():
            train_parts[0].append((usr, like_items_p1))
            test_like_items = copy.deepcopy(like_items_p2)
            if self.filter_flag:
                test_like_items = self.basic.filter_club(like_items_p2, "剧本杀")
            for test_like_item in test_like_items:
                test_parts[0].append([usr, test_like_item])

            train_parts[1].append((usr, like_items_p2))
            test_like_items = copy.deepcopy(like_items_p1)
            if self.filter_flag:
                test_like_items = self.basic.filter_club(like_items_p1, "剧本杀")
            for test_like_item in test_like_items:
                test_parts[1].append([usr, test_like_item])

        for i in range(len(train_parts)):
            train_parts[i] = dict(train_parts[i])

        for i in range(len(num_neigh_thres_range)):
            f_scores.append([])

            for j in range(len(pred_thres_range)):
                f_score_parts = []
                for k in range(len(train_parts)):
                    recommend_data = []

                    user_cf = UserCF(self.database, "user_cf_top_n_recommendation_map.feather",
                                     num_neigh_thres_range[i], pred_thres_range[j],
                                     train_parts[k], True)
                    user_cf.get_top_n()

                    for usr in self.database.like_users:
                        gama = user_cf.run(usr)
                        for recommend_item in gama:
                            recommend_data.append([usr, recommend_item])

                    test_df = pd.DataFrame(data=test_parts[k])
                    test_df = test_df.rename(columns={0: "user", 1: "item"})
                    recommend_df = pd.DataFrame(data=recommend_data)
                    recommend_df = recommend_df.rename(columns={0: "user", 1: "item"})
                    tp_df = pd.merge(test_df, recommend_df, on=["user", "item"])

                    accuracy = len(tp_df) / len(recommend_df)
                    hit_ratio = len(tp_df) / len(test_df)

                    f_score = 2 * accuracy * hit_ratio / (accuracy + hit_ratio)
                    f_score_parts.append(f_score)
                f_scores[i].append(sum(f_score_parts)/len(f_score_parts))
        f_scores = np.array(f_scores)

        result = np.where(f_scores == np.amax(f_scores))
        param_indices = list(zip(result[0], result[1]))[0]
        params = [num_neigh_thres_range[param_indices[0]], pred_thres_range[param_indices[1]]]
        np.save(os.path.join(configs.tuning_result_folder_path, "user_cf_tuning.npy"), f_scores)
        np.save(os.path.join(configs.tuning_result_folder_path, "user_cf_params.npy"), np.asarray(params))
        return params, f_scores

    # 调整itemCF模型的参数
    def itemCF_tuning(self, sim_thres_range, pred_thres_range):
        train_parts = [[], []]
        test_parts = [[], []]
        f_scores = []

        for usr, like_items_p1, like_items_p2 in self.two_fold_partition():
            train_parts[0].append((usr, like_items_p1))
            test_like_items = copy.deepcopy(like_items_p2)
            if self.filter_flag:
                test_like_items = self.basic.filter_club(like_items_p2, "剧本杀")
            for test_like_item in test_like_items:
                test_parts[0].append([usr, test_like_item])

            train_parts[1].append((usr, like_items_p2))
            test_like_items = copy.deepcopy(like_items_p1)
            if self.filter_flag:
                test_like_items = self.basic.filter_club(like_items_p1, "剧本杀")
            for test_like_item in test_like_items:
                test_parts[1].append([usr, test_like_item])

        for i in range(len(train_parts)):
            train_parts[i] = dict(train_parts[i])

        for i in range(len(sim_thres_range)):
            f_scores.append([])

            for j in range(len(pred_thres_range)):
                f_score_parts = []
                for k in range(len(train_parts)):
                    recommend_data = []

                    item_cf = ItemCF(self.database, "item_cf_top_n_recommendation_map.feather",
                                     sim_thres=sim_thres_range[i], pred_thres=pred_thres_range[j],
                                     train_data=train_parts[k], test_flag=True)
                    item_cf.get_top_n()

                    for usr in self.database.like_users:
                        gama = item_cf.run(usr)
                        for recommend_item in gama:
                            recommend_data.append([usr, recommend_item])

                    test_df = pd.DataFrame(data=test_parts[k])
                    test_df = test_df.rename(columns={0: "user", 1: "item"})
                    recommend_df = pd.DataFrame(data=recommend_data)
                    recommend_df = recommend_df.rename(columns={0: "user", 1: "item"})
                    tp_df = pd.merge(test_df, recommend_df, on=["user", "item"])

                    accuracy = len(tp_df) / len(recommend_df)
                    hit_ratio = len(tp_df) / len(test_df)

                    f_score = 2 * accuracy * hit_ratio / (accuracy + hit_ratio)
                    f_score_parts.append(f_score)
                f_scores[i].append(sum(f_score_parts)/len(f_score_parts))
        f_scores = np.array(f_scores)

        result = np.where(f_scores == np.amax(f_scores))
        param_indices = list(zip(result[0], result[1]))[0]
        params = [sim_thres_range[param_indices[0]], pred_thres_range[param_indices[1]]]
        np.save(os.path.join(configs.tuning_result_folder_path, "item_cf_tuning.npy"), f_scores)
        np.save(os.path.join(configs.tuning_result_folder_path, "item_cf_params.npy"), np.asarray(params))
        return params, f_scores

    # 调整HybridCF模型的参数
    def hybridCF_tuning(self, p_range):
        train_parts = [[], []]
        test_parts = [[], []]
        f_scores = []

        for usr, like_items_p1, like_items_p2 in self.two_fold_partition():
            train_parts[0].append((usr, like_items_p1))
            test_like_items = copy.deepcopy(like_items_p2)
            if self.filter_flag:
                test_like_items = self.basic.filter_club(like_items_p2, "剧本杀")
            for test_like_item in test_like_items:
                test_parts[0].append([usr, test_like_item])

            train_parts[1].append((usr, like_items_p2))
            test_like_items = copy.deepcopy(like_items_p1)
            if self.filter_flag:
                test_like_items = self.basic.filter_club(like_items_p1, "剧本杀")
            for test_like_item in test_like_items:
                test_parts[1].append([usr, test_like_item])

        for i in range(len(train_parts)):
            train_parts[i] = dict(train_parts[i])

        for j in range(len(p_range)):
            f_score_parts = []
            for k in range(len(train_parts)):
                recommend_data = []

                hybrid_cf = HybridCF(self.database,
                                     "item_cf_in_hybrid_cf_top_n_recommendation_map.feather",
                                     "user_cf_in_hybrid_cf_top_n_recommendation_map.feather",
                                     p_range[j],
                                     train_data = train_parts[k], test_flag = True)

                for usr in self.database.like_users:
                    gama = hybrid_cf.run(usr)
                    for recommend_item in gama:
                        recommend_data.append([usr, recommend_item])

                test_df = pd.DataFrame(data=test_parts[k])
                test_df = test_df.rename(columns={0: "user", 1: "item"})
                recommend_df = pd.DataFrame(data=recommend_data)
                recommend_df = recommend_df.rename(columns={0: "user", 1: "item"})
                tp_df = pd.merge(test_df, recommend_df, on=["user", "item"])

                accuracy = len(tp_df) / len(recommend_df)
                hit_ratio = len(tp_df) / len(test_df)

                f_score = 2 * accuracy * hit_ratio / (accuracy + hit_ratio)
                f_score_parts.append(f_score)
            f_scores.append(sum(f_score_parts)/len(f_score_parts))
        f_scores = np.array(f_scores)

        result = np.where(f_scores == np.amax(f_scores))
        param_index = result[0][0]
        param = [p_range[param_index]]
        np.save(os.path.join(configs.tuning_result_folder_path, "hybrid_cf_tuning.npy"), f_scores)
        np.save(os.path.join(configs.tuning_result_folder_path, "hybrid_cf_params.npy"), np.asarray(param))
        return param, f_scores

    def store_metrics_of_all_algos(self, num_iter):
        generalized_cf_metrics_data = [[], [], [], []]
        item_cf_metrics_data = [[], [], [], []]
        user_cf_metrics_data = [[], [], [], []]
        hybrid_cf_metrics_data = [[], [], [], []]
        for i in range(num_iter):
            [accuracy, hit_ratio, f_score] = self.calc_metrics(algo_name="generalized_cf")
            coverage = self.calc_coverage(algo_name="generalized_cf")
            generalized_cf_metrics_data[0].append(accuracy)
            generalized_cf_metrics_data[1].append(hit_ratio)
            generalized_cf_metrics_data[2].append(f_score)
            generalized_cf_metrics_data[3].append(coverage)

            [accuracy, hit_ratio, f_score] = self.calc_metrics(algo_name="item_cf")
            coverage = self.calc_coverage(algo_name="item_cf")
            item_cf_metrics_data[0].append(accuracy)
            item_cf_metrics_data[1].append(hit_ratio)
            item_cf_metrics_data[2].append(f_score)
            item_cf_metrics_data[3].append(coverage)

            [accuracy, hit_ratio, f_score] = self.calc_metrics(algo_name="user_cf")
            coverage = self.calc_coverage(algo_name="user_cf")
            user_cf_metrics_data[0].append(accuracy)
            user_cf_metrics_data[1].append(hit_ratio)
            user_cf_metrics_data[2].append(f_score)
            user_cf_metrics_data[3].append(coverage)

            [accuracy, hit_ratio, f_score] = self.calc_metrics(algo_name="hybrid_cf")
            coverage = self.calc_coverage(algo_name="hybrid_cf")
            hybrid_cf_metrics_data[0].append(accuracy)
            hybrid_cf_metrics_data[1].append(hit_ratio)
            hybrid_cf_metrics_data[2].append(f_score)
            hybrid_cf_metrics_data[3].append(coverage)

        generalized_cf_metrics_df = pd.DataFrame(data=generalized_cf_metrics_data)
        generalized_cf_metrics_df["mean"] = generalized_cf_metrics_df.mean(axis=1)
        generalized_cf_metrics_df["std"] = generalized_cf_metrics_df.std(axis=1)
        ic(generalized_cf_metrics_df)

        item_cf_metrics_df = pd.DataFrame(data=item_cf_metrics_data)
        item_cf_metrics_df["mean"] = item_cf_metrics_df.mean(axis=1)
        item_cf_metrics_df["std"] = item_cf_metrics_df.std(axis=1)
        ic(item_cf_metrics_df)

        user_cf_metrics_df = pd.DataFrame(data=user_cf_metrics_data)
        user_cf_metrics_df["mean"] = user_cf_metrics_df.mean(axis=1)
        user_cf_metrics_df["std"] = user_cf_metrics_df.std(axis=1)
        ic(user_cf_metrics_df)

        hybrid_cf_metrics_df = pd.DataFrame(data=hybrid_cf_metrics_data)
        hybrid_cf_metrics_df["mean"] = hybrid_cf_metrics_df.mean(axis=1)
        hybrid_cf_metrics_df["std"] = hybrid_cf_metrics_df.std(axis=1)
        ic(hybrid_cf_metrics_df)

        generalized_cf_metrics_df.to_excel(os.path.join(configs.perf_result_folder_path, "generalized_cf.xlsx"))
        item_cf_metrics_df.to_excel(os.path.join(configs.perf_result_folder_path, "item_cf.xlsx"))
        user_cf_metrics_df.to_excel(os.path.join(configs.perf_result_folder_path, "user_cf.xlsx"))
        hybrid_cf_metrics_df.to_excel(os.path.join(configs.perf_result_folder_path, "hybrid_cf.xlsx"))
