# encoding:utf-8
import random
import sys
from icecream import ic
sys.dont_write_bytecode = True


class Basic(object):

    def __init__(self, database):
        self.database = database

    @staticmethod
    def join_contents_list(a: list, n_a: int, b: list, n_b: int, threshold: int) -> [list, int]:
        """
        将列表a的元素部分或全部加入到列表b的末尾，得到列表c，并去掉重复物品
        :param a: 列表a
        :param n_a: 列表a的元素个数
        :param b: 列表b
        :param n_b: 列表b的元素个数
        :param threshold: 列表c的元素个数阈值
        :return: 列表c和元素个数n_c
        """
        if n_a + n_b <= threshold:
            c = b + a
        else:
            a_ = random.sample(list(a), k=threshold - n_b)
            c = b + a_

        c = list(dict.fromkeys(c))  # 去掉重复元素

        n_c = len(c)

        return [c, n_c]

    def rearrangement(self, beta, n_beta, usr, train_data=None, test_flag=False, filter_flag=True ):
        """
        人工重排算法
        :param beta: 机器排序得到的列表
        :param n_beta: 输出推荐集n_beta的元素个数
        :param usr:
        :param train_data:
        :param test_flag:
        :param filter_flag:
        :return: 列表gama（最终推荐集）和列表gama的元素个数
        """
        m = self.database.get_objs(["club", "动态:推广集", "have", "selected_item"], key="动态")
        if filter_flag:
            m = self.filter_club(m, "剧本杀")
        if test_flag:
            like_items = train_data[str(usr)]
        else:
            like_items = self.database.get_objs(["user", str(usr), 'like', 'item'], key="动态")
        view_items = self.database.get_objs(['user', str(usr), 'view', 'item'], key="动态")
        create_items = self.database.get_objs(['user', str(usr), 'create', 'item'], key="动态")
        m = [i for i in m if (i not in like_items) and (i not in view_items) and (i not in create_items)]
        n_m = len(m)

        thresholds = [0, 45, 50]

        if n_beta == thresholds[0]:
            gama = random.sample(m, k=min(n_m, thresholds[2]))
        elif 0 < n_beta <= thresholds[1]:
            [gama, _] = self.join_contents_list(m, n_m, beta, n_beta, thresholds[2])
        else:
            beta_hat = beta[: thresholds[1]]
            beta_ = beta[thresholds[1]:]
            beta1 = list(set(m) | set(beta_))
            if len(beta1) >= thresholds[2] - thresholds[1]:
                beta1_hat = random.sample(beta1, thresholds[2] - thresholds[1])
            else:
                beta1_hat = beta1
            gama = beta_hat + beta1_hat

        n_gama = len(gama)

        if n_gama > thresholds[2]:
            raise ValueError(f"输出推荐集gama的长度错误，大于{thresholds[2]}")
        return gama, n_gama

    def filter_club(self, items, club_label="剧本杀"):
        """
        对物品进行剧本杀一级CLUB标签的过滤
        """
        item_filtered = []
        for i in items:
            for club in self.database.get_objs(['item', i, 'have', 'club'], key="动态"):
                if club.split(":")[2] == club_label:
                    item_filtered.append(i)
        return item_filtered
