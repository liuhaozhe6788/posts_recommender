# encoding:utf-8
import pandas as pd
import copy
import re
from icecream import ic
import sys

from database import check
from database.preprocessing import to_records
sys.dont_write_bytecode = True


class DataBase:
    def __init__(self, data_file_path=None):
        self.__data = {}
        self.load(data_file_path)
        self.__clubs = list(self.__data.get("club", {}).keys())
        self.__items = list(self.__data.get("item", {}).keys())
        self.__users = list(self.__data.get("user", {}).keys())
        self.__like_users = list(filter(lambda x: self.get_objs(['user', x, 'like', 'item'], key="动态"), list(self.__data.get("user", {}).keys())))

    def load(self, data_file_path):
        if re.search(".*.xlsx$", data_file_path):
            for record in to_records(data_file_path):
                self.add_record(record)

    def get_clubs(self, key="动态", del_prefix: bool = False):
        clubs = list(filter(lambda x: len(x.split(":")) == 4, self.__clubs))
        return self.filter(key, clubs, del_prefix) if key else clubs

    @property
    def data(self):
        import copy
        return copy.deepcopy(self.__data)

    @property
    def clubs(self):
        return self.get_clubs()

    @property
    def items(self):
        return self.__items

    @property
    def users(self):
        return self.__users

    @property
    def like_users(self):
        return self.__like_users

    @property
    def liked_dynamics(self):
        liked_items = list(
            filter(lambda x: self.get_objs(['item', x, 'liked', 'user']), list(self.__data.get("item", {}).keys())))
        return self.filter("动态", liked_items, del_prefix=False)

    def get_objs(self, behavior_record_set: list, key: str = None, del_prefix: bool = False) -> list:
        """从DataBase中获取数据\n
        :param behavior_record_set:行为组合
        :param key: 通过前缀过滤id
        :param del_prefix: 过滤时删除前缀
        :return:
        """
        self.debug("get_objs", behavior_record_set)
        # print(behavior_record_set)
        subj, subj_id, behavior, obj = behavior_record_set

        subj_id = str(subj_id)
        # print(subj_id)
        m1 = check.behavior_info((subj, behavior, obj))["multiple"]
        subj_id = check.item_id_check(subj_id, check.item_categories) if subj == "item" \
            else check.club_id_check(subj_id, check.item_categories) if subj == "club" else subj_id
        r = copy.deepcopy(self.__data.get(subj, {}).get(subj_id, {}).get((behavior, obj), [] if m1 else None))
        return self.filter(key, r, del_prefix) if m1 and key else r

    def add_record(self, behavior_record: list):
        """向DataBase中添加数据\n
        :param behavior_record:
        :return:
        """
        self.debug("add_record", behavior_record)
        subj, subj_id, behavior, obj, obj_id = behavior_record
        try:
            subj_id = str(subj_id)
            obj_id = str(obj_id)
            subj_id = check.item_id_check(subj_id, check.item_categories) if subj == "item" \
                else check.club_id_check(subj_id, check.item_categories) if subj == "club" else subj_id
            obj_id = check.item_id_check(obj_id, check.item_categories) if obj == "item" \
                else check.club_id_check(obj_id, check.item_categories) if obj == "club" else obj_id

        except AssertionError as e:
            print("AssertionWarning:", e)
        else:
            self.pop_record(behavior_record)  # 删除无效历史数据
            if True:
                m1, r1 = list(check.behavior_info((subj, behavior, obj)).values())
                d1 = self.__data.setdefault(subj, {}).setdefault(subj_id, {})
                d1[(behavior, obj)] = (d1.get((behavior, obj), []) + [obj_id]) if m1 else obj_id

            if r1:
                m2, _ = list(check.behavior_info(r1).values())
                d2 = self.__data.setdefault(r1[0], {}).setdefault(obj_id, {})
                d2[(r1[1], r1[2])] = (d2.get((r1[1], r1[2]), []) + [subj_id]) if m2 else subj_id

    def pop_record(self, behavior_record: list):
        """从DataBase中删除数据\n
        :param behavior_record:
        :return:
        """
        self.debug("pop_record", behavior_record)
        subj, subj_id, behavior, obj, obj_id = behavior_record
        try:
            subj_id = str(subj_id)
            obj_id = str(obj_id)
            subj_id = check.item_id_check(subj_id, check.item_categories) if subj == "item" \
                else check.club_id_check(subj_id, check.item_categories) if subj == "club" else subj_id
            obj_id = check.item_id_check(obj_id, check.item_categories) if obj == "item" \
                else check.club_id_check(obj_id, check.item_categories) if obj == "club" else obj_id
        except AssertionError as e:
            print("AssertionWarning:", e)  # 处理item_id_check和club_id_check里的AssertionError
        else:
            if True:
                m1, r1 = list(check.behavior_info((subj, behavior, obj)).values())
                d1 = self.__data.get(subj, {}).get(subj_id, {})
                if not m1 and d1.get((behavior, obj), None):
                    d1.pop((behavior, obj))
                elif m1 and obj_id in (l1 := d1.get((behavior, obj), [])):
                    l1.pop(l1.index(obj_id))
            if r1:
                m2, _ = list(check.behavior_info(r1).values())
                d2 = self.__data.get(r1[0], {}).get(obj_id, {})
                if not m2 and d2.get((behavior, obj), None):
                    d2.pop((r1[1], r1[2]))
                elif m2 and subj_id in (l2 := d2.get((r1[1], r1[2]), [])):
                    l2.pop(l2.index(subj_id))

    @staticmethod
    def filter(key: str, data: list[str], del_prefix: bool = False):
        if key[-1] == ":":
            key = key[:-1]
        try:
            check.category(key, check.item_categories)
        except AssertionError as e:
            print("AssertionWarning:", e)
        else:
            k = len(key)
            data = list(map(lambda s: s[k+1:] if del_prefix else s, filter(lambda s: s[:k] == key, data)))
        finally:
            return data

    @staticmethod
    def debug(*args):
        debug_ = False
        if debug_:
            print(args)


if __name__ == "__main__":
    # 测试
    import os
    import configs

    data_base = DataBase(os.path.join(configs.data_folder_path, "data_20220222.xlsx"))

    '''
    for k1, v1 in data_base.data.items():
        print(k1)
        for k2, v2 in v1.items():
            for k3, v3 in v2.items():
                print('|  |--{}: {}'.format(k3, v3))
    '''

    ic(data_base.get_objs(['user', 48479, 'follow', 'user']))  # 48447,48536,49171,48449
    ic(data_base.get_objs(['user', 1, 'like', 'item'], key="动态"))
    ic(data_base.get_objs(['user', 51044, 'join', 'club'], key="动态"))
    # ic(data_base.get_objs(['club', '动态:二级标签:游戏电竞', 'have', 'selected_item'], key="动态"))
#    ic(set(map(lambda club: club.split(":")[-1].split("_")[0], list(filter(lambda s: s.startswith("动态"), data_base.clubs)))))
    ic(data_base.get_objs(['item', '动态:10355', 'have', 'image_url']))
    ic(data_base.get_objs(['item', '动态:10355', 'have', 'club'], key="动态"))
    ic(data_base.clubs)
    # ic(len(data_base.users))
    # ic(len(data_base.like_users))
    # ic(data_base.get_objs(['user', 48502, 'comment', 'item']))
    # ic(data_base.get_objs(['item', '评论:133', 'have', 'source']))
    # ic(data_base.get_objs(['user', 49070, 'like', 'item'], key="动态"))
    # ic(len(data_base.liked_dynamics))
    m = data_base.get_objs(["club", "动态:二级标签:剧本杀:剧本杀", "include", "item"], key="动态")
    ic(m)
