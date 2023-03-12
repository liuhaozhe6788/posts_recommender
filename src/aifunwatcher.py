# encoding:utf-8
import os
import sys
from PyQt5.QtWidgets import QApplication
from qt_visual import UI

import utils
import configs
import database
sys.dont_write_bytecode = True

if __name__ == "__main__":
    # 显示GUI
    utils.create_folder_paths()
    app = QApplication(sys.argv)
    window = UI(database.DataBase(os.path.join(configs.data_folder_path, "data_20220222.xlsx")))
    window.show()
    app.exec_()