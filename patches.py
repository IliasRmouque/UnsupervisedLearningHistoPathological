import csv
import os
import time
from math import sqrt

# import psutil
from random import randint, sample

import numpy as np
import imageio
from PIL import Image
from openslide import ImageSlide, open_slide
from os import listdir, mkdir, wait
from utils.path import IMG_DIR, PTCH_DIR, get_ALL_PATCHES_DIR, List_IMAGES, ptch_size, LOD
import sys
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None
IMG_SIZE=64


BACKGROUND_PATCH=190
BLACK_AREA = 10


class Patch:
    def __init__(self, name, size, row, column):
        self.imgName = name
        self.row = row
        self.column = column
        self.size = size
        self.term= ""
        self.img = None

        #TODO: Find a way to initialise properly
        self.colour = -1

    def get_pos(self):
        return self.column * self.size, self.row * self.size



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
            img = ImageSlide(IMG_DIR + name)
            f+=cutPatches(img, name)
    np.save("./patchesArray", f)

def cut_all_patches(img, sourcename):
    w, h = img.dimensions
    n_samples= (h // IMG_SIZE * w // IMG_SIZE)
    cutPatches(img, sourcename, n_samples)

def show_colors(IMG_DIR):
    fileNameList = listdir(IMG_DIR)
    lc=[]
    i=0
    for f in fileNameList:
        i+=1
        if i > 2000: break
        img= np.asarray(Image.open(IMG_DIR+f))
        img= np.mean(img, axis=(0, 1))
        X = img[0]
        Y = img[1]
        Z = img[2]

        p = 1.04
        #print(X**2+Y**2+Z**2-((X+Y+Z)**2)/3)
        if True: #500<(X**2)+(Y**2)+(Z**2)-((X+Y+Z)**2)/3 and Z <(np.power(np.subtract(X*np.cos(p), Y*np.sin(p)), 2))/-40+(X*np.sin(p)+Y*np.cos(p))-10 :
                lc.append(img)
                # plt.imshow(np.asarray(Image.open(IMG_DIR+f)))
                # plt.show()
                # print('(', X, ',', Y, ",", Z, "),")

    lc= np.asarray(lc)
    # print(lc.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(lc[:, 0], lc[:, 1],lc[:, 2], c=lc/255)
    from scipy.linalg import norm
    X = np.arange(1, 256, 10)
    Y = np.arange(1, 256, 10)
    X, Y = np.meshgrid(X, Y)
    p = 1.04
    Z = (-0.5*(X**2) -0.5*(Y**2)+X*Y+ 35*X+35*Y)/49.5
    # origin = np.array([0, 0, 0])
    # # axis and radius
    # p0 = np.array([1, 1, 1])*255
    # p1 = np.array([0 ,0, 0])
    # R = 22
    # # vector in direction of axis
    # v = p1 - p0
    # # find magnitude of vector
    # mag = norm(v)
    # # unit vector in direction of axis
    # v = v / mag
    # # make some vector not in the same direction as v
    # not_v = np.array([1, 0, 0])
    # if (v == not_v).all():
    #     not_v = np.array([0, 1, 0])
    # # make vector perpendicular to v
    # n1 = np.cross(v, not_v)
    # # normalize n1
    # n1 /= norm(n1)
    # # make unit vector perpendicular to v and n1
    # n2 = np.cross(v, n1)
    # # surface ranges over t from 0 to length of axis and 0 to 2*pi
    # t = np.linspace(0, mag, 100)
    # theta = np.linspace(0, 2 * np.pi, 100)
    # # use meshgrid to make 2d arrays
    # t, theta = np.meshgrid(t, theta)
    # # generate coordinates for surface
    # X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    # ax.plot_surface(X, Y, Z)
    # from matplotlib import cm
    ax.plot_surface(X, Y, Z, linewidth=1, antialiased=False)
    #ax.plot_surface(Z, X, Y, linewidth=1, antialiased=False)
    # ax.plot_surface(Y, X, Z, linewidth=1, antialiased=False)
    ax.set_ylim(0, 255)
    ax.set_xlim(0, 255)
    ax.set_zlim(0, 255)

    plt.show()




def cutPatches(img, source_img_name, number_of_samples=1000):
    """
    This function divide the images into patches of size IMG_SIZE ans save them as a png image named after the name of the image and the coordinates of the pixel at the top left of that region.
    :param img: the image that as to be processed
    :param number_of_samples: the wanted number of sample
    :param source_img_name: name of
    :return:an array with all the numpy array corresponding to the chosen patches (might be slightly less than the number of required samples depending on image quality)
    """
    show_colors(IMG_DIR)
    exit(0)
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

                    print('patch ', cpt, '/', number_of_samples)
                patch.close()
                if cpt == number_of_samples:
                    break
        if cpt == number_of_samples:
            break
    print("saved patches:", cpt, " \n loss:",  1-(cpt)/number_of_samples)
    return nArrayRes

def get_file_name_list():
    pass


def clear_patches(current_dir):
    print(current_dir)
    for files in listdir(current_dir):
        print(current_dir +'/'+ files)
        if os.path.isdir(current_dir +'/'+ files) and files[-5:] == "tiles":
            patches_dir = current_dir +'/'+ files
            break


    lfiles =[]
    for files in listdir(patches_dir):
        img = np.asarray(Image.open(patches_dir +'/'+files))
        img = np.mean(img, axis=(0, 1))
        X = img[0]
        Y = img[1]
        Z = img[2]
        p = 1.04
        if (Z > (np.power(np.subtract(X * np.cos(p), Y * np.sin(p)), 2)) / -40 + (X * np.sin(p) + Y * np.cos(p))) or not (Y < (X + Z - 10) / 2.1):
                os.remove(patches_dir +'/'+ files)
        else:
                lfiles.append(files)

    listTile=[]
    with open(current_dir + "/tile_selection.tsv") as f:
        rcsv = csv.reader(f, delimiter="\t")
        # read the first line that holds column labels
        csv_labels = rcsv.__next__()
        for record in rcsv:
            if record[3] == '1' and record[0] + ".png" in lfiles :
                listTile.append(record)

        with open(current_dir + "/tile_selection2.tsv", 'w') as f2:
            wcsv= csv.writer(f2, delimiter="\t")
            wcsv.writerow(csv_labels)
            for record in listTile:
                wcsv.writerow(record)
    f = {}
    for name in listdir(patches_dir):
        if name[-4:] == ".png":
            img = imageio.imread(patches_dir +'/'+ name)
            f[name] = img.copy()

    np.save(current_dir + '/patchesArray.npy', f)


def makes_all_patches(lod, size):
    for i in range(3):
        dir = get_ALL_PATCHES_DIR(lod, size)
        ld = listdir(dir)
        list_patch = sample(ld, min(20000, len(ld)) )
        f = {}
        for name in list_patch:
            if name[-4:] == ".png":
                img = imageio.imread(dir + '/' + name)
                f[name] = np.asarray(img.copy())
        np.save(dir + '/patchesArray' +str(i) + '.npy', f)


def makes_all_patches_treshold(lod, size):
    for i in range(3):
        dir = get_ALL_PATCHES_DIR(lod, size)
        ld = listdir(dir)
        list_patch = sample(ld, min(20000, len(ld)) )
        f = {}
        c = 0

        for name in list_patch:
            c += 1
            print(c, '/', len(ld))
            if name[-4:] == ".png":
                img = Image.open(dir + '/' + name)
                img = img.convert("L")
                img = img.point(lambda p: p > 115 and 255)
                f[name] = np.asarray(to_3d(np.asarray(img)))
        np.save(dir + '/patchesArray_treshold' +str(i) + '.npy', f)

def to_3d(img):
    res=[]
    for i in img:
        res.append([])
        for j in i:
            res[-1].append([j])
    return res


def makes_img_ptch_treshold(dir):
        ld = listdir(dir)
        f = {}
        i=0
        for name in ld:
            i+=1
            print(i, '/', len(ld))
            if name[-4:] == ".png":
                img = Image.open(dir + '/' + name)
                img = img.convert("L")
                img = img.point(lambda p: p > 115 and 255)
                f[name] = np.asarray(to_3d(np.asarray(img)))
        np.save(dir + '../patchesArray_treshold'+ '.npy', f)


if __name__ == '__main__':
    import sys


    for IMG_ID, IMG_NAME in List_IMAGES:
        IMG_DIR = './../Images/Patches/' + IMG_NAME + '_' + str(ptch_size) \
                  + "_LOD=" + str(LOD) + '/' + IMG_NAME + '_tiles/'
        makes_img_ptch_treshold(IMG_DIR)
    makes_all_patches_treshold(8, 128)
    print("\n\n\n\n\n\n\n\n\n\n\n")
    #clear_patches("/home/rmouque/Bureau/Images/Patches/A17-7275_-_2019-11-06_15.25.13_128_LOD=8")
    #clear_patches(sys.argv[1])



    # for dname in listdir(IMG_DIR + '../../'):
    #     if "H11002875" in dname:
    #         cpt=0
    #
    #
    # # pass
    # for dname in listdir(IMG_DIR + '../../'):
    #     fileNameList = listdir(IMG_DIR + '../../'+dname +'/' +dname[:32] + '_tiles' +'/')
    #     f = {}
    #     g = {}
    #     print(len(fileNameList))
    #     if True: #not os.path.isfile(IMG_DIR + '../../'+ dname +'/patchesArray.npy'):
    #         for name in fileNameList:
    #             if name[-4:] == ".png":
    #                 img = imageio.imread(IMG_DIR + '../../'+dname +'/' +dname[:32] + '_tiles' +'/'+ name)
    #                 f[name] = img.copy()
    #
    #
    #         np.save(IMG_DIR + '../../'+ dname +'/patchesArray.npy', f)


