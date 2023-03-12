# encoding:utf-8
import os
import random
import numpy as np
import itertools
import sys

sys.dont_write_bytecode = True

import configs
import utils
from database import DataBase
from .item_cf import ItemCF
from .user_cf import UserCF
from .basic import Basic

np.set_printoptions(threshold=np.inf)


class HybridCF():

    def __init__(self, database, item_cf_buffer_name, user_cf_buffer_name, p=0.6, train_data=None, test_flag=False, filter_flag=True):
        self.database = database
        self.p = p  # 混合比例参数user_cf的结果/item_cf的结果
        self.item_cf_buffer_name = item_cf_buffer_name
        self.user_cf_buffer_name = user_cf_buffer_name
        self.train_data = train_data
        self.test_flag = test_flag
        self.filter_flag = filter_flag
        self.user_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "user_cf_params.npy"))
        self.item_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "item_cf_params.npy"))
        self.user_cf = UserCF(self.database, self.user_cf_buffer_name, self.user_cf_params[0], self.user_cf_params[1], self.train_data, self.test_flag, self.filter_flag)
        self.item_cf = ItemCF(self.database, self.item_cf_buffer_name, self.item_cf_params[0], self.item_cf_params[1], self.train_data, self.test_flag, self.filter_flag)
        self.basic = Basic(database)

    def get_top_n(self):
        self.user_cf.get_top_n()
        self.item_cf.get_top_n()

    def run(self, usr):
        max_length_of_rec = 50
        user_cf_items = self.user_cf.run(usr)
        num_user_cf_items = len(user_cf_items)
        k = int(max_length_of_rec * self.p)
        user_cf_items = user_cf_items[: min(k, num_user_cf_items)]
        num_user_cf_items = len(user_cf_items)

        item_cf_items = self.item_cf.run(usr)
        num_item_cf_items = len(item_cf_items)
        item_cf_items = item_cf_items[: min(max_length_of_rec-k, num_item_cf_items)]
        num_item_cf_items = len(item_cf_items)

        gama = list(itertools.chain(*zip(user_cf_items, item_cf_items)))

        if num_user_cf_items <= num_item_cf_items:
            gama += item_cf_items[num_user_cf_items: ]
        else:
            gama += user_cf_items[num_item_cf_items: ]

        gama = list(dict.fromkeys(gama))
        gama = gama[: min(len(gama), max_length_of_rec)]

        return self.database.filter("动态", gama, del_prefix=False)
