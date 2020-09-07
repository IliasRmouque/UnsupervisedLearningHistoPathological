import numpy as np
from openslide import OpenSlide
from os import listdir, mkdir, wait
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG_SIZE=64
IMG_DIR="../images/entrainementColon/"
BACKGROUND_PATCH=210
BLACK_AREA = 10

def dirTreating():
    try:
        mkdir('./patches/')
        print("creating the 'patches' folder")
    except FileExistsError:
        print("the 'patches' folder already exists")
    fileNameList = listdir(IMG_DIR)
    f = []
    for name in fileNameList:
        if name[-4:] == "ndpi":
            img = OpenSlide(IMG_DIR + name)
            f+=cutPatches(img, name)
    np.save("./patchesArray", f)

def fromIMGToArray(img):
    pass


def cutPatches(img, fileName, number_of_samples=1000):
    """
    This function divide the images into patches of size IMG_SIZE ans save them as a png image named after the name of the image and the coordinates of the pixel at the top left of that region.
    :param img:
    :param fileName:
    :return:an array with all the numpy array corresponding to the kept patches
    """
    w, h = img.dimensions
    print(fileName + " width=" + str(w) + " height=" + str(h) + "number of patches:" + str(h//IMG_SIZE*w//IMG_SIZE))
    allMean = []
    color = []
    greens = []
    blues = []
    reds = []
    nArrayRes=[]
    cpt=0
    cptloss=0
    numStop= round((h//IMG_SIZE*w//IMG_SIZE)/number_of_samples)
    for i in  range(0,h//IMG_SIZE,1):
        print("row ", i, "/", h//IMG_SIZE)
        for j in range(w//IMG_SIZE):
            if numStop > 0:
                numStop -= 1
            else:
                cptloss += 1
                ptch= img.read_region(location=(int(j*IMG_SIZE),int(i*IMG_SIZE)), level=0, size=(IMG_SIZE,IMG_SIZE))
                temp= np.asarray(ptch)
                npimg= np.delete(temp, 3, 2)

                allMean = np.mean(np.mean(npimg, 0), 0)
                if allMean[1]<(allMean[0]+allMean[2] -10)/2.1:
                    cpt+=1
                    numStop = round((h // IMG_SIZE * w // IMG_SIZE) / number_of_samples)
                    ptch.save("./patches/"+fileName+"_x:"+str(i*h//IMG_SIZE)+"_y:"+str(j*w//IMG_SIZE)+".png", "PNG")
                    nArrayRes.append(npimg)
    print("saved patches:",cpt," \n loss:",  1-(cpt)/number_of_samples)
    return nArrayRes



if __name__ == '__main__':
    """img = OpenSlide("../images/entrainementColon/A17-4822_-_2019-11-06_14.45.09.ndpi")
    cutPatches(img, "test")"""
    dirTreating()

