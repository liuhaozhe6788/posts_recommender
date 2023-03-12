# encoding:utf-8
import os
import numpy as np
import pandas as pd
import requests
import io
import shutil
from icecream import ic
import sys

import configs
import plotting
from database.preprocessing import clean_data
sys.dont_write_bytecode = True


def cleaned_data_vis(cleaned_data_file):
    """
    对清洗的数据进行可视化
    :param cleaned_data_file: 清洗的数据的文档
    :return: None
    """
    cleaned_data_path = os.path.join(configs.data_folder_path, cleaned_data_file)
    user_df, item_df, club_df = clean_data(cleaned_data_path)

    if os.path.exists(configs.visualization_folder_path):
        shutil.rmtree(configs.visualization_folder_path)
    os.mkdir(configs.visualization_folder_path)

    plotgenerator = plotting.PlotGenerator()

    # print(user_df.head())
    # print(item_df.head())
    # print(club_df.head())

    # ic(user_df[(user_df["subject_id"] == 49070) & (user_df["behavior"] == "like")])

    # 用户数量、物品数量、点赞用户数量、不存在图片信息的剧本杀动态数量、点赞行为的数量、用户-物品点赞行为的稀疏度
    user_behaviors_df = user_df[user_df["behavior"].isin(["follow", "like", "view", "create", "join", "comment"])]
    users = user_behaviors_df["subject_id"].unique()
    ic(len(users))

    dynamics_item_df = item_df[item_df["content_type"].isin(["动态"])]
    ic(len(dynamics_item_df))

    script_kill_item_df = dynamics_item_df[dynamics_item_df["item_desc"].str.startswith("剧本杀")]
    ic(len(script_kill_item_df))

    user_like_df = user_df[user_df["behavior"].isin(["like"])]
    like_users = user_like_df["subject_id"].unique()
    ic(len(like_users))
    liked_items = user_like_df["object_id"].unique()
    ic(len(liked_items))

    dynamics_item_df_ = dynamics_item_df.copy()
    dynamics_item_df_ = dynamics_item_df_.rename(columns={"item_id": "object_id"})
    ic(dynamics_item_df_.columns, user_like_df.columns)
    liked_script_kill_item_df = pd.merge(dynamics_item_df_[["object_id", "item_desc"]], user_like_df, on="object_id")
    liked_script_kill_item_df = liked_script_kill_item_df[liked_script_kill_item_df["item_desc"].str.startswith("剧本杀")]
    liked_script_kill_item_df = liked_script_kill_item_df.drop_duplicates(subset=["object_id"])
    liked_script_kill_items = liked_script_kill_item_df["object_id"].unique()
    ic(len(liked_script_kill_items))

    dynamics_item_df_ = dynamics_item_df.copy()
    dynamics_item_without_info_df_ = dynamics_item_df_[(dynamics_item_df_["image_url"] == "*") & (dynamics_item_df_["video_url"] == "*")]
    ic(len(dynamics_item_without_info_df_))  # 1个,但该动态未出现在用户行为数据中

    ic(len(user_like_df))

    data_sparsity = len(user_like_df)/(len(like_users)*len(liked_items))
    ic(data_sparsity)

    # 动态中所有剧本杀club二级标签的数量分布，countplot
    script_kill_item_df_ = script_kill_item_df.copy()
    club_two_desc = script_kill_item_df_.apply(lambda x: x["item_desc"].split(":")[1], axis=1)
    club_two_desc = club_two_desc.to_frame().rename(columns={0: "desc"})
    # ic(club_one_desc)
    plotgenerator.count_plot(x="desc",
                             data=club_two_desc,
                             figwidth=40,
                             figheight=20,
                             new_xlabel="剧本杀CLUB二级标签",
                             new_ylabel="动态的数量",
                             new_title="所有""剧本杀""动态在CLUB二级标签的数量分布",
                             new_fig_name=os.path.join(configs.visualization_folder_path, f"num_of_items_per_club2_dist.png"),
                             savefig=True
                             )

    # 用户点赞动态中所有剧本杀club二级标签的数量分布，countplot
    liked_script_kill_item_df_ = liked_script_kill_item_df.copy()
    club_two_desc = liked_script_kill_item_df_.apply(lambda x: x["item_desc"].split(":")[1], axis=1)
    club_two_desc = club_two_desc.to_frame().rename(columns={0: "desc"})
    # ic(club_one_desc)
    plotgenerator.count_plot(x="desc",
                             data=club_two_desc,
                             figwidth=40,
                             figheight=20,
                             new_xlabel="剧本杀CLUB二级标签",
                             new_ylabel="动态的数量",
                             new_title="用户点赞剧本杀动态在CLUB二级标签的数量分布",
                             new_fig_name=os.path.join(configs.visualization_folder_path, f"num_of_like_items_per_club2_dist.png"),
                             savefig=True
                             )

    # 每个用户点赞物品的数量分布，histplot
    num_like_df = user_like_df.groupby(["subject_id"]).count().sort_values(by="behavior", ascending=False).reset_index()
    num_like_df[["subject_id", "behavior"]].to_excel(os.path.join(configs.data_folder_path, "like_users.xlsx"), index=False)
    plotgenerator.hist_plot(x=num_like_df["behavior"],
                            figwidth=30,
                            figheight=20,
                            binwidth=5,
                            new_xlabel="用户点赞次数",
                            new_ylabel="用户相同点赞次数出现的数量",
                            new_title="用户点赞次数直方图",
                            new_fig_name=os.path.join(configs.visualization_folder_path, f"num_of_likes_per_user_dist.png"),
                            savefig=True
                            )

    # 每个物品被点赞的用户数量分布
    num_liked_df = user_like_df.groupby(["object_id"]).count().sort_values(by="behavior", ascending=False).reset_index()
    plotgenerator.hist_plot(x=num_liked_df["behavior"],
                            figwidth=30,
                            figheight=20,
                            binwidth=1,
                            new_xlabel="动态被点赞次数",
                            new_ylabel="动态被点赞次数出现的数量",
                            new_title="动态被点赞次数直方图",
                            new_fig_name=os.path.join(configs.visualization_folder_path, f"num_of_likes_per_item_dist.png"),
                            savefig=True
                            )

if __name__ == "__main__":
    cleaned_data_vis("data_20220222.xlsx")