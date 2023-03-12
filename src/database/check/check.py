import os
import pandas as pd
from icecream import ic
import sys
sys.dont_write_bytecode = True

behavior_file_path = os.path.realpath(os.path.join(__file__, "../behavior.xlsx"))

behaviors = dict(
    (
        tuple(row["behavior"].split(";")), {
            "multiple": row["multiple"],
            "reverse": None if pd.isna(row["reverse"]) else tuple(row["reverse"].split(";"))
        }) for idx, row in pd.read_excel(behavior_file_path, "behavior").iterrows()
)

item_categories = dict(
    (
        row["item_category"], str(row["club_categories"]).split(",")
    ) for idx, row in pd.read_excel(behavior_file_path, "item_category_data").iterrows()
)

def category(cat: str, d: dict):
    for cat in (c_ := [c.strip() for c in cat.split(":")]):
        if isinstance(d, list) and cat in d:
            d = None
        elif isinstance(d, dict) and cat in d:
            d = d.get(cat)
        else:
            raise AssertionError("\"{}\" - 不存在的物品/标签种类".format(cat))
    return ":".join(c_)


def behavior_info(behavior_combination: tuple, /):
    info = behaviors.get(behavior_combination, False)
    if not info:
        raise AssertionError("\"{} {} {}\" - 不是一个正常的行为".format(behavior_combination[0], behavior_combination[1],
                                                               behavior_combination[2]))
    return info


def item_id_check(item_id: str, item_cats: dict):
    item_id_ = [v.strip() for v in item_id.split(":")]
    if len(item_id_) == 2:
        category(item_id_[0], item_cats)
    else:
        raise AssertionError("\"{}\" - 格式错误，无法解析".format(item_id))
    return ":".join(item_id_)


def club_id_check(club_id, item_cats: dict):
    club_id_ = [v.strip() for v in club_id.split(":")]
    if len(club_id_) == 4:
        category("{}:{}".format(*club_id_[:2]), item_cats)
    elif len(club_id_) == 2 and club_id_[1] == "推广集":
        category(club_id_[0], item_cats)
    else:
        raise AssertionError("\"{}\" - 格式错误，无法解析".format(club_id))
    return ":".join(club_id_)


if __name__ == "__main__":
    # print(behavior_file_path)
    # print(f"所有行为：{behaviors}")
    # print(f"所有物品类型：{item_categories}")
    # print(behavior_info(('user', 'follow', 'user')))
    # print(item_id_check("动态:2", item_categories))
    print(club_id_check("动态:推广集", item_categories))
    print(club_id_check("动态:二级标签:2:3", item_categories))
