# encoding:utf-8
import os
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys

sys.dont_write_bytecode = True

import configs
import utils
import seaborn as sns
from database import DataBase
from item_cf import ItemCF
from basic import Basic

np.set_printoptions(threshold=np.inf)


class SVDCF(ItemCF):

    def __init__(self, database, buffer_name=None, train_data=None, test_flag=False, filter_flag=True):

        super().__init__(database, buffer_name, train_data, test_flag, filter_flag)

    def run(self):
        ui_matrix = self.create_ui_matrix()
        print(ui_matrix.shape)
        U, sigma, Vt = np.linalg.svd(ui_matrix)
        ax = sns.lineplot(data=sigma, palette="dark")
        plt.show()



if __name__ == "__main__":
    utils.create_folder_paths()
    mydb = DataBase(os.path.join(configs.data_folder_path, "data_20220222.xlsx"))
    svd_cf = SVDCF(mydb)
    svd_cf.run()
