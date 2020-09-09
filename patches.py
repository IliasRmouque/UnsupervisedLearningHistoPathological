import numpy as np
from openslide import OpenSlide
from os import listdir, mkdir, wait
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG_SIZE=64
IMG_DIR="../../images/entrainementColon/"
PTCH_DIR="./patchesImageFull/"

BACKGROUND_PATCH=210
BLACK_AREA = 10

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\n"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    count = 45



def dirTreating():
    try:
        mkdir(PTCH_DIR)
        print("creating the 'patches' folder")
    except FileExistsError:
        print("the " + PTCH_DIR + " folder already exists")
    fileNameList = listdir(IMG_DIR)
    f = []
    for name in fileNameList:
        if name[-4:] == "ndpi":
            img = OpenSlide(IMG_DIR + name)
            f+=cutPatches(img, name)
    np.save("./patchesArray", f)

def cut_all_patches(img, sourcename):
    w, h = img.dimensions
    n_samples= (h // IMG_SIZE * w // IMG_SIZE)
    cutPatches(img, sourcename, n_samples)

def cutPatches(img, source_img_name, number_of_samples=1000):
    """
    This function divide the images into patches of size IMG_SIZE ans save them as a png image named after the name of the image and the coordinates of the pixel at the top left of that region.
    :param img: the image that as to be processed
    :param number_of_samples: the wanted number of sample
    :param source_img_name:
    :return:an array with all the numpy array corresponding to the chosen patches (might be slightly less than the number of required samples depending on image quality)
    """
    # for the progress bar
    unitColor = '\033[5;36m\033[5;47m'
    endColor = '\033[0;0m\033[0;0m'

    w, h = img.dimensions
    print(source_img_name + " width=" + str(w) + " height=" + str(h) + "number of patches:" + str(h//IMG_SIZE*w//IMG_SIZE))
    nArrayRes=[]
    cpt=0
    cptloss=0
    numStop= round((h//IMG_SIZE*w//IMG_SIZE)/number_of_samples)
    for i in  range(0,h//IMG_SIZE,1):
        #print("row ", i, "/", h//IMG_SIZE)
        for j in range(w//IMG_SIZE):
            if numStop > 0:
                numStop -= 1
            else:
                fileName = source_img_name + "_x:" + str(j*IMG_SIZE) + "_y:" + str( h - i * IMG_SIZE) + ".png"
                patch = img.read_region(location=(int(j * IMG_SIZE), int(i * IMG_SIZE)), level=0,
                                        size=(IMG_SIZE, IMG_SIZE))
                temp = np.asarray(patch)
                cptloss += 1
                npimg = np.delete(temp, 3, 2)
                allMean = np.mean(np.mean(npimg, 0), 0)
                if allMean[1] < (allMean[0] + allMean[2] - 10) / 2.1:
                    cpt += 1
                    numStop = round(((((h // IMG_SIZE)-i) * (w // IMG_SIZE) - j) / (number_of_samples - cpt + 1)))
                    patch.save(PTCH_DIR+fileName, "PNG")
                    nArrayRes.append((fileName, npimg))
                    printProgressBar(cpt,number_of_samples)
                    print('patch ', cpt, '/', number_of_samples)
                patch.close()
                if cpt == number_of_samples:
                    break
        if cpt == number_of_samples:
            break
    print("saved patches:", cpt, " \n loss:",  1-(cpt)/number_of_samples)
    return nArrayRes



if __name__ == '__main__':
    img = OpenSlide(IMG_DIR + "A17-4822_-_2019-11-06_14.45.09.ndpi")
    w, h = img.dimensions
    img.get_thumbnail((h,w)).save("test", "PNG")
