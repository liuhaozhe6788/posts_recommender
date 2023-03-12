# encoding:utf-8
import os
import shutil
import sys

import configs
sys.dont_write_bytecode = True


def create_folder_paths():
    """
    创建图片结果的文件夹路径
    :return:
    """
    if not os.path.exists(configs.algo_result_folder_path):
        os.mkdir(configs.algo_result_folder_path)

    if not os.path.exists(configs.tuning_result_folder_path):
        os.mkdir(configs.tuning_result_folder_path)

    if not os.path.exists(configs.rec_result_folder_path):
        os.mkdir(configs.rec_result_folder_path)

    if not os.path.exists(configs.perf_result_folder_path):
        os.mkdir(configs.perf_result_folder_path)

    if not os.path.exists(configs.visualization_folder_path):
        os.mkdir(configs.visualization_folder_path)

    if not os.path.exists(configs.buffer_files_folder_path):
        os.mkdir(configs.buffer_files_folder_path)
