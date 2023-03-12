# encoding:utf-8
import os
import pandas as pd
import numpy as np
import sys

import configs
import database
import visualization
from algos import AlgosOperator, ItemCF, UserCF, HybridCF
import utils
sys.dont_write_bytecode = True


if __name__ == "__main__":
    utils.create_folder_paths()

    # 清洗后的数据可视化
    # visualization.cleaned_data_vis("data_20220222.xlsx")

    mydb = database.DataBase(os.path.join(configs.data_folder_path, "data_20220222.xlsx"))
    algos = AlgosOperator(mydb)
    # params, itemCF_fscores = algos.itemCF_tuning(np.arange(0.5, 0.8, 0.03), np.arange(0, 0.2, 0.04))
    # params, userCF_fscores = algos.userCF_tuning(np.arange(20, 50, 3), np.arange(0.1, 0.14, 0.01))
    # param, hybridCF_fscores = algos.hybridCF_tuning(np.arange(0.4, 0.6, 0.02))

    # 调参结果可视化
    # visualization.itemCF_tuning_vis(np.arange(0.5, 0.8, 0.03), np.arange(0, 0.2, 0.04))
    # visualization.userCF_tuning_vis(np.arange(20, 50, 3), np.arange(0.1, 0.14, 0.01))
    # visualization.hybridCF_tuning_vis(np.arange(0.4, 0.6, 0.02))

    # iteration = 20
    # arr = np.zeros(iteration, dtype=float)
    # for i in range(iteration):
    #     arr[i] = algos.calc_train_test_ratio()
    # print(np.average(arr))

    # 计算和存储性能指标
    # algos.store_metrics_of_all_algos(num_iter=50)

    # 给某用户运行单个推荐算法
    # item_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "item_cf_params.npy"))
    # item_cf = ItemCF(mydb,
    #                  "item_cf_top_n_recommendation_map.feather",
    #                  item_cf_params[0],
    #                  item_cf_params[1]
    #                  )
    # item_cf.get_top_n()
    # print(item_cf.run("49070"))
    #
    # user_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "user_cf_params.npy"))
    # user_cf = UserCF(mydb,
    #                  "user_cf_top_n_recommendation_map.feather",
    #                  user_cf_params[0],
    #                  user_cf_params[1]
    #                  )
    # user_cf.get_top_n()
    # print(user_cf.run("49070"))
    #
    # hybrid_cf = HybridCF(mydb,
    #                      "item_cf_in_hybrid_cf_top_n_recommendation_map.feather",
    #                      "user_cf_in_hybrid_cf_top_n_recommendation_map.feather",
    #                      )
    # print(hybrid_cf.run("49070"))

    # 推荐结果可视化
    algo_names = ["generalized_cf", "item_cf", "user_cf", "hybrid_cf"]
    legend_names = ["Hybrid Generalized CF", "Item-based CF", "User-based CF", "Naive Hybrid CF"]
    gama_dfs = []
    for i in range(len(algo_names)):
        gama_df = algos.run_all_users(algo_names[i])
        visualization.club_dist_vis(algo_names[i], legend_names[i], gama_df, mydb)
        gama_dfs.append(gama_df)
    gama_results = dict(zip(algo_names, gama_dfs))
    visualization.hit_vis(algo_names, legend_names, gama_results, mydb)
