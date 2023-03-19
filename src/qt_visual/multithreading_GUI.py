# encoding: utf-8
import sys
import numpy as np
import os
import logging
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from .image_widget import ImageWidget
from .video_widget import VideoWidget
import configs
import database
import utils
from algos import GeneralizedCF, ItemCF, UserCF, HybridCF

sys.dont_write_bytecode = True
logging.basicConfig(format="%(message)s", level=logging.INFO)

N = 0  # 计数已结束的线程个数


class WorkerSignals(QObject):
    # 定义Widget之间的信号
    finish = pyqtSignal(bytes)  # 一幅新图片加载完成的信号
    finish_all = pyqtSignal(bool)  # 所有图片加载完成的信号


class Runnable(QRunnable):
    def __init__(self, scroll_widget, n, n_threads):
        self.widget = scroll_widget
        self.n = n
        self.n_threads = n_threads
        self.signals = WorkerSignals()
        super().__init__()

    # GUI界面的更新不能运行在子线程中
    def run(self):
        global N
        frame_data = self.widget.add_img()
        N = N + 1
        logging.info(f"thread No.{self.n} is finished, {N}/{self.n_threads} threads are finished")
        self.signals.finish.emit(frame_data)  # 图片下载完后，发出信号把图片比特流传给给GUI，用于界面的更新，显示图片

        if N == self.n_threads:
            logging.info(f"all {N} contents downloaded")
            self.signals.finish_all.emit(True)  # 所有图片加载完后，发出True信号，让GUI上的确认按钮重新生效


class UI(QWidget):
    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        uic.loadUi(os.path.join(configs.qt_visual_folder_path, "rec_result.ui"), self)  # 导入.ui文件

        utils.create_folder_paths()

        self.db = db
        self.generalized_cf = GeneralizedCF(self.db)
        item_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "item_cf_params.npy"))
        self.item_cf = ItemCF(self.db, "item_cf_top_n_recommendation_map.feather", item_cf_params[0], item_cf_params[1])
        self.item_cf.get_top_n()
        user_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "user_cf_params.npy"))
        self.user_cf = UserCF(self.db, "user_cf_top_n_recommendation_map.feather", user_cf_params[0], user_cf_params[1])
        self.user_cf.get_top_n()
        hybrid_cf_params = np.load(os.path.join(configs.tuning_result_folder_path, "hybrid_cf_params.npy"))
        self.hybrid_cf = HybridCF(self.db,
                                  "item_cf_in_hybrid_cf_top_n_recommendation_map.feather",
                                  "user_cf_in_hybrid_cf_top_n_recommendation_map.feather",
                                  hybrid_cf_params[0])
        self.hybrid_cf.get_top_n()

        # 增加用户ID输入框
        self.yes_btn = QPushButton()
        self.yes_btn.setFixedHeight(30)
        self.yes_btn.setIcon(QIcon(os.path.join(configs.qt_img_folder_path, "yes.jpeg")))
        self.yes_btn.setIconSize(QSize(30, 30))
        self.line_edit = QLineEdit()
        self.line_edit.setFixedHeight(30)
        self.uid_label = QLabel("用户ID:")
        self.uid_label.setFont(QFont("Sanserif", 20))
        self.uid_label.setFixedHeight(30)

        self.inputWidget = QWidget()
        self.inputLayout = QHBoxLayout()

        self.scrollWindow = QScrollArea()
        self.scrollWindowWidget = QWidget()
        self.scrollWindowLayout = QVBoxLayout()

        self.scrollLike = QScrollArea()  # 滚动区域
        self.scrollLikeWidget = QWidget()  # 滚动区域内的Widget
        self.scrollLikeLayout = QHBoxLayout()  # 滚动区域内Widget的layout

        self.scrollAlgo_1 = QScrollArea()
        self.scrollAlgo_1Widget = QWidget()
        self.scrollAlgo_1Layout = QHBoxLayout()

        self.scrollAlgo_2 = QScrollArea()
        self.scrollAlgo_2Widget = QWidget()
        self.scrollAlgo_2Layout = QHBoxLayout()

        self.scrollAlgo_3 = QScrollArea()
        self.scrollAlgo_3Widget = QWidget()
        self.scrollAlgo_3Layout = QHBoxLayout()

        self.scrollAlgo_4 = QScrollArea()
        self.scrollAlgo_4Widget = QWidget()
        self.scrollAlgo_4Layout = QHBoxLayout()

        self.uid_result = QLabel("用户ID: ")
        self.uid_result.setFont(QFont("Sanserif", 15))
        self.uid_result.setFixedHeight(30)
        self.like_prompt = QLabel("由于您已点赞了如下动态: ")
        self.like_prompt.setFont(QFont("Sanserif", 15))
        self.like_prompt.setFixedHeight(30)
        self.generalized_cf_prompt = QLabel("Hybrid Generalized CF给您推荐如下动态: ")
        self.generalized_cf_prompt.setFont(QFont("Sanserif", 15))
        self.generalized_cf_prompt.setFixedHeight(30)
        self.item_cf_prompt = QLabel("Item-based CF给您推荐如下动态: ")
        self.item_cf_prompt.setFont(QFont("Sanserif", 15))
        self.item_cf_prompt.setFixedHeight(30)
        self.user_cf_prompt = QLabel("User-based CF给您推荐如下动态: ")
        self.user_cf_prompt.setFont(QFont("Sanserif", 15))
        self.user_cf_prompt.setFixedHeight(30)
        self.hybrid_cf_prompt = QLabel("Naive Hybrid CF给您推荐如下动态: ")
        self.hybrid_cf_prompt.setFont(QFont("Sanserif", 15))
        self.hybrid_cf_prompt.setFixedHeight(30)

        scrollArea_height = 650

        self.scrollLikeWidget.setLayout(self.scrollLikeLayout)
        self.scrollLike.setWidgetResizable(True)
        self.scrollLike.setFixedHeight(scrollArea_height)
        self.scrollLike.setWidget(self.scrollLikeWidget)

        self.scrollAlgo_1Widget.setLayout(self.scrollAlgo_1Layout)
        self.scrollAlgo_1.setWidgetResizable(True)
        self.scrollAlgo_1.setFixedHeight(scrollArea_height)
        self.scrollAlgo_1.setWidget(self.scrollAlgo_1Widget)

        self.scrollAlgo_2Widget.setLayout(self.scrollAlgo_2Layout)
        self.scrollAlgo_2.setWidgetResizable(True)
        self.scrollAlgo_2.setFixedHeight(scrollArea_height)
        self.scrollAlgo_2.setWidget(self.scrollAlgo_2Widget)

        self.scrollAlgo_3Widget.setLayout(self.scrollAlgo_3Layout)
        self.scrollAlgo_3.setWidgetResizable(True)
        self.scrollAlgo_3.setFixedHeight(scrollArea_height)
        self.scrollAlgo_3.setWidget(self.scrollAlgo_3Widget)

        self.scrollAlgo_4Widget.setLayout(self.scrollAlgo_4Layout)
        self.scrollAlgo_4.setWidgetResizable(True)
        self.scrollAlgo_4.setFixedHeight(scrollArea_height)
        self.scrollAlgo_4.setWidget(self.scrollAlgo_4Widget)

        self.scrollWindowLayout.addWidget(self.uid_result)
        self.scrollWindowLayout.addWidget(self.like_prompt)
        self.scrollWindowLayout.addWidget(self.scrollLike)
        self.scrollWindowLayout.addWidget(self.generalized_cf_prompt)
        self.scrollWindowLayout.addWidget(self.scrollAlgo_1)
        self.scrollWindowLayout.addWidget(self.item_cf_prompt)
        self.scrollWindowLayout.addWidget(self.scrollAlgo_2)
        self.scrollWindowLayout.addWidget(self.user_cf_prompt)
        self.scrollWindowLayout.addWidget(self.scrollAlgo_3)
        self.scrollWindowLayout.addWidget(self.hybrid_cf_prompt)
        self.scrollWindowLayout.addWidget(self.scrollAlgo_4)

        self.scrollWindowWidget.setLayout(self.scrollWindowLayout)
        # self.scrollWindow.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollWindow.setWidgetResizable(True)
        # self.scrollWindow.setFixedHeight(1050)
        self.scrollWindow.setWidget(self.scrollWindowWidget)

        self.inputWidget.setLayout(self.inputLayout)
        self.inputLayout.addWidget(self.uid_label)
        self.inputLayout.addWidget(self.line_edit)
        self.inputLayout.addWidget(self.yes_btn)

        self.vLayout = QVBoxLayout()
        self.vLayout.addWidget(self.inputWidget)
        self.vLayout.addWidget(self.scrollWindow)
        self.setLayout(self.vLayout)

        self.scrollWidgets = []

        self.yes_btn.clicked.connect(self.yes_btn_signal)

    def yes_btn_signal(self):
        print("yes button clicked")
        self.yes_btn.setEnabled(False)
        global N
        N = 0
        uid = self.line_edit.text()
        self.uid_result.setText(f"用户ID: {uid}")
        self.delete_all_res()
        self.scrollWidgets = []
        like_items = self.db.get_objs(['user', uid, 'like', 'item'], key="动态")
        like_items.reverse()
        # n_items_max = 30
        # like_items = like_items[: min(n_items_max, len(like_items))]

        for like_item in like_items:
            like_item_club = ",".join((list(filter(lambda x: ":".join(x.split(":")[-2:]), self.db.get_objs(['item', like_item, 'have', 'club'], key="动态")))))
            if like_item_url := self.db.get_objs(['item', like_item, 'have', 'image_url']):
                self.add_widgets(
                    "img",
                    like_item_url[0],
                    like_item.split(":")[-1],
                    ":".join(like_item_club.split(":")[-2:]),
                    self.scrollLikeLayout
                )
            elif like_item_url := self.db.get_objs(['item', like_item, 'have', 'video_url']):
                self.add_widgets(
                    "vid",
                    like_item_url,
                    like_item.split(":")[-1],
                    ":".join(like_item_club.split(":")[-2:]),
                    self.scrollLikeLayout
                )

        algo_1_items = self.generalized_cf.run(uid)
        for algo_1_item in algo_1_items:
            algo_1_item_club = ",".join((list(filter(lambda x: ":".join(x.split(":")[-2:]), self.db.get_objs(['item', algo_1_item, 'have', 'club'], key="动态")))))
            if algo_1_item_url := self.db.get_objs(['item', algo_1_item, 'have', 'image_url']):
                self.add_widgets(
                    "img",
                    algo_1_item_url[0],
                    algo_1_item.split(":")[-1],
                    ":".join(algo_1_item_club.split(":")[-2:]),
                    self.scrollAlgo_1Layout
                )
            elif algo_1_item_url := self.db.get_objs(['item', algo_1_item, 'have', 'video_url']):
                self.add_widgets(
                    "vid",
                    algo_1_item_url,
                    algo_1_item.split(":")[-1],
                    ":".join(algo_1_item_club.split(":")[-2:]),
                    self.scrollAlgo_1Layout
                )

        algo_2_items = self.item_cf.run(uid)
        for algo_2_item in algo_2_items:
            algo_2_item_club = ",".join((list(filter(lambda x: ":".join(x.split(":")[-2:]), self.db.get_objs(['item', algo_2_item, 'have', 'club'], key="动态")))))
            if algo_2_item_url := self.db.get_objs(['item', algo_2_item, 'have', 'image_url']):
                self.add_widgets(
                    "img",
                    algo_2_item_url[0],
                    algo_2_item.split(":")[-1],
                    ":".join(algo_2_item_club.split(":")[-2:]),
                    self.scrollAlgo_2Layout
                )
            elif algo_2_item_url := self.db.get_objs(['item', algo_2_item, 'have', 'video_url']):
                self.add_widgets(
                    "vid",
                    algo_2_item_url,
                    algo_2_item.split(":")[-1],
                    ":".join(algo_2_item_club.split(":")[-2:]),
                    self.scrollAlgo_2Layout
                )

        algo_3_items = self.user_cf.run(uid)
        for algo_3_item in algo_3_items:
            algo_3_item_club = ",".join((list(filter(lambda x: ":".join(x.split(":")[-2:]), self.db.get_objs(['item', algo_3_item, 'have', 'club'], key="动态")))))
            if algo_3_item_url := self.db.get_objs(['item', algo_3_item, 'have', 'image_url']):
                self.add_widgets(
                    "img",
                    algo_3_item_url[0],
                    algo_3_item.split(":")[-1],
                    ":".join(algo_3_item_club.split(":")[-2:]),
                    self.scrollAlgo_3Layout
                )
            elif algo_3_item_url := self.db.get_objs(['item', algo_3_item, 'have', 'video_url']):
                self.add_widgets(
                    "vid",
                    algo_3_item_url,
                    algo_3_item.split(":")[-1],
                    ":".join(algo_3_item_club.split(":")[-2:]),
                    self.scrollAlgo_3Layout
                )

        algo_4_items = self.hybrid_cf.run(uid)
        for algo_4_item in algo_4_items:
            algo_4_item_club = ",".join((list(filter(lambda x: ":".join(x.split(":")[-2:]), self.db.get_objs(['item', algo_4_item, 'have', 'club'], key="动态")))))
            if algo_4_item_url := self.db.get_objs(['item', algo_4_item, 'have', 'image_url']):
                self.add_widgets(
                    "img",
                    algo_4_item_url[0],
                    algo_4_item.split(":")[-1],
                    ":".join(algo_4_item_club.split(":")[-2:]),
                    self.scrollAlgo_4Layout
                )
            elif algo_4_item_url := self.db.get_objs(['item', algo_4_item, 'have', 'video_url']):
                self.add_widgets(
                    "vid",
                    algo_4_item_url,
                    algo_4_item.split(":")[-1],
                    ":".join(algo_4_item_club.split(":")[-2:]),
                    self.scrollAlgo_4Layout
                )

        self.add_contents()

    def add_widgets(self, content_type_, url_, itemid_, club_, scroll_layout_):
        if content_type_ == "img":
            img_widget = ImageWidget(url_, itemid_, club_)
            scroll_layout_.addWidget(img_widget)
            self.scrollWidgets.append(img_widget)

        elif content_type_ == "vid":
            vid_widget = VideoWidget(url_, itemid_, club_)
            scroll_layout_.addWidget(vid_widget)
            self.scrollWidgets.append(vid_widget)
        else:
            return

    def add_contents(self):
        """
        使用线程池加载图片
        """
        pool = QThreadPool.globalInstance()
        logging.info(f"Running {len(self.scrollWidgets)} Threads")
        for i in range(len(self.scrollWidgets)):
            runnable = Runnable(self.scrollWidgets[i], i, len(self.scrollWidgets))
            runnable.signals.finish.connect(self.scrollWidgets[i].updateUI)
            runnable.signals.finish_all.connect(lambda x: self.yes_btn.setEnabled(x))
            pool.start(runnable)

    def delete_all_res(self):
        """
        将历史图片信息都抹去
        """
        for i in reversed(range(self.scrollLikeLayout.count())):
            self.scrollLikeLayout.itemAt(i).widget().setParent(None)

        for i in reversed(range(self.scrollAlgo_1Layout.count())):
            self.scrollAlgo_1Layout.itemAt(i).widget().setParent(None)

        for i in reversed(range(self.scrollAlgo_2Layout.count())):
            self.scrollAlgo_2Layout.itemAt(i).widget().setParent(None)

        for i in reversed(range(self.scrollAlgo_3Layout.count())):
            self.scrollAlgo_3Layout.itemAt(i).widget().setParent(None)

        for i in reversed(range(self.scrollAlgo_4Layout.count())):
            self.scrollAlgo_4Layout.itemAt(i).widget().setParent(None)


app = QApplication(sys.argv)
window = UI(database.DataBase(os.path.join(configs.data_folder_path, "data_20220222.xlsx")))
window.show()
app.exec_()
