# encoding:utf-8
import os

import numpy as np
import pandas as pd
from icecream import ic
import sys
sys.dont_write_bytecode = True

import configs
import plotting


def itemCF_tuning_vis(sim_thres_range, pred_thres_range):
    my_plot = plotting.PlotGenerator()
    f_scores = np.load(os.path.join(configs.tuning_result_folder_path, "item_cf_tuning.npy"))
    df = pd.DataFrame({
        "sim_thres": np.tile(sim_thres_range, pred_thres_range.size),
        "f_score": f_scores.flatten("F"),
        "pred_thres": np.repeat(pred_thres_range, sim_thres_range.size)
    })
    my_plot.line_plot(x="sim_thres",
                      y="f_score",
                      hue="pred_thres",
                      data=df,
                      figwidth=24,
                      figheight=20,
                      ylim_low=0.0255,
                      ylim_high=0.0295,
                      legend_fontsize=30,
                      xtick_fontsize=40,
                      ytick_fontsize=40,
                      xtick_rot=0,
                      xlabel_fontsize=50,
                      ylabel_fontsize=50,
                      title_fontsize=60,
                      new_legend_labels=[f'预测值阈值={"%.2f" % i}' for i in pred_thres_range],
                      new_xlabel="相似度阈值",
                      new_ylabel="F1-score",
                      new_title=f"Item-based CF模型的调参结果",
                      new_fig_name=os.path.join(configs.tuning_result_folder_path, f"item_cf_tuning.png"),
                      savefig=True
                      )


def userCF_tuning_vis(num_neigh_thres_range, pred_thres_range):
    my_plot = plotting.PlotGenerator()
    f_scores = np.load(os.path.join(configs.tuning_result_folder_path, "user_cf_tuning.npy"))
    df = pd.DataFrame({
        "num_neigh_thres": np.tile(num_neigh_thres_range, pred_thres_range.size),
        "f_score": f_scores.flatten("F"),
        "pred_thres": np.repeat(pred_thres_range, num_neigh_thres_range.size)
    })
    my_plot.line_plot(x="num_neigh_thres",
                      y="f_score",
                      hue="pred_thres",
                      data=df,
                      figwidth=24,
                      figheight=20,
                      ylim_low=0.02855,
                      ylim_high=0.02876,
                      legend_fontsize=30,
                      xtick_fontsize=40,
                      ytick_fontsize=40,
                      xtick_rot=0,
                      xlabel_fontsize=50,
                      ylabel_fontsize=50,
                      title_fontsize=60,
                      new_legend_labels=[f'预测值阈值={"%.2f" % i}' for i in pred_thres_range],
                      new_xlabel="近邻个数",
                      new_ylabel="F1-score",
                      new_title=f"User-based CF模型的调参结果",
                      new_fig_name=os.path.join(configs.tuning_result_folder_path, f"user_cf_tuning.png"),
                      savefig=True
                      )

def hybridCF_tuning_vis(p_range):
    my_plot = plotting.PlotGenerator()
    f_scores = np.load(os.path.join(configs.tuning_result_folder_path, "hybrid_cf_tuning.npy"))
    df = pd.DataFrame({
        "p": p_range,
        "f_score": f_scores,
    })
    my_plot.line_plot(x="p",
                      y="f_score",
                      data=df,
                      figwidth=24,
                      figheight=20,
                      ylim_low=0.031,
                      ylim_high=0.0325,
                      xtick_fontsize=40,
                      ytick_fontsize=40,
                      xtick_rot=0,
                      xlabel_fontsize=50,
                      ylabel_fontsize=50,
                      title_fontsize=60,
                      new_xlabel="混合比例",
                      new_ylabel="F1-score",
                      new_title=f"Naive Hybrid CF模型的调参结果",
                      new_fig_name=os.path.join(configs.tuning_result_folder_path, f"hybrid_cf_tuning.png"),
                      savefig=True
                      )


if __name__ == "__main__":
    itemCF_tuning_vis(np.arange(0.5, 0.8, 0.03), np.arange(0, 0.2, 0.04))
    userCF_tuning_vis(np.arange(20, 50, 3), np.arange(0.1, 0.14, 0.01))
    hybridCF_tuning_vis(np.arange(0.4, 0.6, 0.02))
    user_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "user_cf_params.npy"))
    item_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "item_cf_params.npy"))
    hybrid_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "hybrid_cf_params.npy"))
    ic(user_cf_params)
    ic(item_cf_params)
    ic(hybrid_cf_params)

    user_cf_tuning_res = np.load(os.path.join(configs.tuning_result_folder_path, "user_cf_tuning.npy"))
    item_cf_tuning_res = np.load(os.path.join(configs.tuning_result_folder_path, "item_cf_tuning.npy"))
    hybrid_cf_tuning_res = np.load(os.path.join(configs.tuning_result_folder_path, "hybrid_cf_tuning.npy"))
    ic(np.max(user_cf_tuning_res))
    ic(np.max(item_cf_tuning_res))
    ic(np.max(hybrid_cf_tuning_res))