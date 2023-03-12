# encoding: utf-8
import sys
import os
import requests
from .video_widget import VideoWidget

sys.dont_write_bytecode = True


class ImageWidget(VideoWidget):

    def __init__(self, url=None, itemID=None, club=None):
        super().__init__(url, itemID, club)

    def add_img(self):
        # time.sleep(1)
        image_data = None
        while image_data is None:
            image_data = requests.get(self.url).content
        return image_data
