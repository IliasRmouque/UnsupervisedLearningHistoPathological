import os
from utils.path import RAW_IMG_DIR


def raw_images_to_patch(size, LODs):
    fileNameList = os.listdir(RAW_IMG_DIR)
    for name in fileNameList:

