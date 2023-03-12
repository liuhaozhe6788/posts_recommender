# encoding:utf-8
import os

import numpy as np
import pandas as pd
from icecream import ic
import sys
sys.dont_write_bytecode = True

import configs
from algos import AlgosOperator
import database
import plotting
import utils
from algos import Basic


def _filter_club_labels(clubs, club_label="剧本杀"):
    return [i for i in clubs if i.split(":")[2] == club_label]


def _freq_calc(items, clubs_list, n_clubs, data_base_) -> (np.ndarray, list, list):
    """
    计算某个物品集的出现的CLUB标签频率
    :param items:某个物品集
    :param clubs_list:CLUB标签列表
    :param n_clubs:CLUB标签总个数
    :param data_base_:DataBase类的实例
    :return: CLUB标签在某个物品集出现的频率
    """
    frequencies = np.zeros(n_clubs, dtype=float)
    for it in items:
        # clubs_of_item = list(map(lambda x: ":".join(x.split(":")[:-1]), data_base_.get_objs(['item', it, 'have', 'club'], key="动态")))
        clubs_of_item = _filter_club_labels(data_base_.get_objs(['item', it, 'have', 'club'], key="动态"))
        for c in clubs_of_item:  # 一个动态可能属于多个标签
            # club_label = c.split(":")[2]
            frequencies[clubs_list.index(c)] += 1 / len(clubs_of_item)
    frequencies /= frequencies.sum()

    return frequencies


def hit_vis(algo_names, legend_names, gama_results, data_base_):
    """
    所有CLUB标签的推荐精度计算与可视化
    一个CLUB标签c的推荐命中率的计算：
    推荐集中CLUB标签c出现次数最高且点赞物品中CLUB标签c出现次数最高的所有用户数/点赞物品中CLUB标签c出现次数最高的所有用户数
    :param algo_names:所有算法名称
    :param legend_names:图表上legend的对应名称
    :param gama_results:所有用户的推荐结果
    :param data_base_:DataBase类的实例
    :return: None
    """
    # clubs_list = list(dict.fromkeys(list(map(lambda x: x.split(":")[-2], list(filter(lambda s: s.startswith("动态"), data_base_.clubs))))))
    clubs_list = _filter_club_labels(data_base_.clubs)
    n_clubs = len(clubs_list)
    all_algos_accuracies = []

    for algo in algo_names:
        like_occurence = np.zeros(n_clubs, dtype=int)
        hit_occurence = np.zeros(n_clubs, dtype=int)
        gama_result = gama_results[algo]
        for usr in data_base_.like_users:
            user_like = data_base_.get_objs(['user', usr, 'like', 'item'], key="动态")
            if Basic(data_base_).filter_club(user_like):
                user_like_freq = _freq_calc(user_like, clubs_list, n_clubs, data_base_)
                user_like_club_max = np.where(user_like_freq == np.amax(user_like_freq))
                like_occurence[user_like_club_max] += 1

                gama = list(gama_result[gama_result["user"] == str(usr)]["recommendation"])[0]
                gama_freq = _freq_calc(gama, clubs_list, n_clubs, data_base_)
                gama_club_max = np.where(gama_freq == np.amax(gama_freq))
                hit_occurence[np.intersect1d(user_like_club_max, gama_club_max)] += 1

        nonzero_ids = like_occurence.nonzero()
        like_occurence = like_occurence[nonzero_ids]
        hit_occurence = hit_occurence[nonzero_ids]

        hit = np.zeros(len(like_occurence), dtype=float)
        np.divide(hit_occurence, like_occurence, hit)
        ic(like_occurence, hit_occurence, hit)
        all_algos_accuracies.append(hit)
    clubs_list = list(np.array(clubs_list)[nonzero_ids])

    df = pd.DataFrame({
        "x_values": clubs_list * len(algo_names),
        "freq": np.concatenate(all_algos_accuracies, axis=None),
        "cat": [item for item in algo_names for _ in range(len(hit))]
    })

    my_plot = plotting.PlotGenerator()

    # print(df)

    my_plot.bar_plot(x="x_values",
                     y="freq",
                     hue="cat",
                     data=df,
                     figwidth=40,
                     figheight=20,
                     legend_fontsize="40",
                     xtick_fontsize=40,
                     ytick_fontsize=40,
                     xtick_rot=0,
                     xlabel_fontsize=50,
                     ylabel_fontsize=50,
                     title_fontsize=60,
                     new_xticks=list(map(lambda x: x[:5] + "\n" + x[5:] if len(x) > 5 else x, list(map(lambda x: x.split(":")[-1], clubs_list)))),
                     # new_xticks=list(map(lambda x: x.split(":")[-1], clubs_list)),
                     new_legend_labels=[f'算法{i}的推荐性能' for i in legend_names],
                     new_xlabel="CLUB标签",
                     new_ylabel="CLUB标签命中率",
                     new_title=f"所有推荐算法的推荐性能",
                     new_fig_name=os.path.join(configs.perf_result_folder_path, f"hit_results.png"),
                     savefig=True
                     )
    return None


def club_dist_vis(algo_name, title_name, gama_result, data_base_):
    """
    点赞数最多的前3位用户的点赞物品与推荐集在CLUB标签分布的可视化
    :param algo_name:算法名称
    :param title_name:标题的算法名称
    :param gama_result:所有用户的推荐结果
    :param data_base_:DataBase类的实例
    :return: None
    """
    # clubs_list = list(dict.fromkeys(list(map(lambda x: x.split(":")[-2], list(filter(lambda s: s.startswith("动态"), data_base_.clubs))))))
    clubs_list = _filter_club_labels(data_base_.clubs)
    n_clubs = len(clubs_list)

    # selected_users = random.choices(data_base_.users, k=3)
    selected_users = [49070, 48449, 48493]

    my_plot = plotting.PlotGenerator()
    dfs = []
    new_titles = []
    subplot_index = ['(a)', '(b)', '(c)']
    for k in range(len(selected_users)):
        user_view = data_base_.get_objs(['user', selected_users[k], 'view', 'item'], key="动态")
        n_user_view = len(user_view)
        n_recently_view = 300
        n_recently_view = min(n_recently_view, n_user_view)

        user_like = data_base_.get_objs(['user', selected_users[k], 'like', 'item'], key="动态")
        n_user_like = len(user_like)
        n_recently_like = 300
        n_recently_like = min(n_recently_like, n_user_like)

        if n_recently_view > 0:
            user_view_freq = _freq_calc(user_view[-n_recently_view:], clubs_list, n_clubs, data_base_)

        if n_recently_like > 0:
            user_like_freq = _freq_calc(user_like[-n_recently_like:], clubs_list, n_clubs, data_base_)

        gama = list(gama_result[gama_result["user"] == str(selected_users[k])]["recommendation"])[0]
        gama_freq = _freq_calc(gama, clubs_list, n_clubs, data_base_)

        df = pd.DataFrame({
            "x_values": clubs_list * 2,
            "freq": np.concatenate((user_like_freq, gama_freq), axis=None),
            "cat": ["user_like_freq"] * len(user_like_freq) + ["gama_freq"] * len(user_like_freq)
        })
        # print(df)
        dfs.append(df)
        new_titles.append(f"{subplot_index[k]} 算法{title_name}的结果中用户{selected_users[k]}的推荐动态与点赞动态的比较")
    my_plot.bar_plots(x="x_values",
                      y="freq",
                      hue="cat",
                      data=dfs,
                      figwidth=40,
                      figheight=20*len(selected_users),
                      legend_fontsize="50",
                      xtick_fontsize=45,
                      ytick_fontsize=45,
                      xtick_rot=0,
                      xlabel_fontsize=50,
                      ylabel_fontsize=50,
                      title_fontsize=65,
                      new_xticks=list(map(lambda x: x[:5] + "\n" + x[5:] if len(x) > 5 else x,
                                          list(map(lambda x: x.split(":")[-1], clubs_list)))),
                      # new_xticks=list(map(lambda x: x.split(":")[-1], clubs_list)),
                      new_legend_labels=[f'用户点赞动态中{n_clubs}个club标签的频率', f'推荐集中{n_clubs}个club标签的频率'],
                      new_xlabel="CLUB标签",
                      new_ylabel="频率",
                      new_titles=new_titles,
                      new_fig_name=os.path.join(configs.perf_result_folder_path, f"{algo_name}_selected_results.png"),
                      savefig=True
                      )
    return None


if __name__ == "__main__":
    utils.create_folder_paths()

    mydb = database.DataBase(os.path.join(configs.data_folder_path, "data_20220222.xlsx"))
    myalgoOperator = AlgosOperator(mydb)

    algo_names = ["generalized_cf", "item_cf", "user_cf", "hybrid_cf"]
    legend_names = ["Hybrid Generalized CF", "Item-based CF", "User-based CF", "Naive Hybrid CF"]
    gama_dfs = []
    for i in range(len(algo_names)):
        gama_df = myalgoOperator.run_all_users(algo_names[i])
        club_dist_vis(algo_names[i], legend_names[i], gama_df, mydb)
        gama_dfs.append(gama_df)
    gama_results = dict(zip(algo_names, gama_dfs))
    hit_vis(algo_names, legend_names, gama_results, mydb)
