# encoding: utf-8
import sys
sys.dont_write_bytecode = True
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2


class VideoWidget(QWidget):

    def __init__(self, url=None, itemID=None, club=None):
        super().__init__()
        self.contentType = "vid"
        self.url = url
        self.itemID = itemID
        self.club = club
        self.snapLabel = QLabel()
        self.clubLabel = QLabel(f"{self.itemID}\n{self.club}")
        self.layout = QVBoxLayout()
        self.height = 500
        self.fontSize = 15

        self.clubLabel.setFont(QFont("Sanserif", self.fontSize))

        self.setLayout(self.layout)
        self.layout.addWidget(self.snapLabel)
        self.layout.addWidget(self.clubLabel)

    def add_img(self):
        # time.sleep(1)
        frame = None
        while frame is None:
            cap = cv2.VideoCapture(self.url)

            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_MSEC, fps / 20)

            ret, frame = cap.read()
        frame_data = cv2.imencode('.png', frame)[1].tostring()
        return frame_data

    def updateUI(self, frame_data):
        image = QImage()
        image.loadFromData(frame_data)
        pix_map = QPixmap(image)
        self.snapLabel.setPixmap(pix_map.scaled(int(pix_map.width() / pix_map.height() * self.height), self.height))
        self.snapLabel.setFixedWidth(int(pix_map.width() / pix_map.height() * self.height))

        self.clubLabel.setFont(QFont("Sanserif", self.fontSize))
        self.clubLabel.setFixedWidth(int(pix_map.width() / pix_map.height() * self.height))
        self.setFixedWidth(int(pix_map.width() / pix_map.height() * self.height))

    @property
    def content_type(self):
        return self.contentType
