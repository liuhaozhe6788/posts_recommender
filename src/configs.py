# encoding:utf-8
import os.path
import sys
sys.dont_write_bytecode = True

root_folder_path = os.path.abspath(os.path.join(__file__, ".."))
root_folder_path = os.path.abspath(os.path.join(root_folder_path, ".."))
# print(root_folder_path)

src_folder_path = os.path.join(root_folder_path, "src")
data_folder_path = os.path.join(root_folder_path, "data")
qt_img_folder_path = os.path.join(root_folder_path, "img")
qt_visual_folder_path = os.path.join(src_folder_path, "qt_visual")
buffer_files_folder_path = os.path.join(root_folder_path, "buffer")
visualization_folder_path = os.path.join(root_folder_path, "visualization_results")
algo_result_folder_path = os.path.join(root_folder_path, "algo_results")

tuning_result_folder_path = os.path.join(algo_result_folder_path, "tuning_result")
rec_result_folder_path = os.path.join(algo_result_folder_path, "rec_result")
perf_result_folder_path = os.path.join(algo_result_folder_path, "perf_result")


__all__ = ["src_folder_path", "data_folder_path", "qt_img_folder_path", "qt_visual_folder_path", "buffer_files_folder_path", "visualization_folder_path", "algo_result_folder_path",
           "tuning_result_folder_path", "perf_result_folder_path", "rec_result_folder_path"]
# 当使用from config import * 时，只能export这些路径
