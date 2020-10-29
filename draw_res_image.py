from PIL import Image
from utils.path import IMG_DIR, PTCH_DIR
import numpy as np
from numba import njit
from math import ceil


def initiate_color_masks(nb_clust, size):
    colorMask=[Image.new('RGB', size=(size, size), color="hsl("+str(round(i/nb_clust*360))+ ",100%, 50%)") for i in range(nb_clust)]
    colorMask.append(Image.new('RGB', size=(size, size), color="hsl(0, 0%, 0%)"))
    return colorMask

def initiate_color_masks_gpu(nb_clust, size):
    colorMask=[np.asarray(Image.new('RGB', size=(size, size), color="hsl("+str(round(i/nb_clust*360))+ ",100%, 50%)")) for i in range(nb_clust)]
    return colorMask


def draw_res_image(img_size, patch_dic, folder= IMG_DIR, nb_clust=10, downscale_factor= 1):

    ptch_size =list(patch_dic.values())[0].size
    img = Image.new('RGB', size=(ceil(img_size[0]/downscale_factor), ceil(img_size[1]/downscale_factor)), color=(255,255,255))
    #TODO: patch_dir.values[0].size ?
    colorMask= initiate_color_masks(nb_clust, size=ceil(ptch_size/downscale_factor))

    cpt=0


    for k, ptc in patch_dic.items():
        try:
            cpt+=1
            patch = Image.open(folder + k + ".png")
            if downscale_factor != 1:
                patch.thumbnail((ceil(ptch_size/downscale_factor), ceil(ptch_size/downscale_factor)))
            patch = Image.blend(patch, colorMask[ptc.colour], 0.5)
            img.paste(patch, (ceil(ptc.get_pos()[0]/downscale_factor), ceil(ptc.get_pos()[1]/downscale_factor)))
            patch.close()
        except:
            pass
            #print(ptc.colour, "  ",  ptc.get_pos,"   ",k )

    px_legend = 10
    for i, col in enumerate(colorMask):
        for j in range(px_legend):
            img.paste(col, (ceil(j*ptch_size/downscale_factor), ceil(i*ptch_size/downscale_factor)))


    print(cpt)


    return img


def draw_origin_image(img_size, patch_dir,  nb_clust=10):

    ptch_size =list(patch_dir.values())[0].size
    img = Image.new('RGB', size=img_size, color=(255,255,255))
    cpt=0
    for k, ptc in patch_dir.items():
        cpt+=1
        patch = Image.open(PTCH_DIR + k + ".png")
        img.paste(patch, ptc.get_pos())
        patch.close()
    print(cpt)

    return img

@njit
def go_faster(img , patch,x, y):
    img[x][y]=patch


def draw_res_image_gpu(img_size, patch_dir,  nb_clust=10):
    ptch_size =list(patch_dir.values())[0].size
    img = np.asarray(Image.new('RGB', size=img_size, color=(255,255,255)), dtype= np.uint8)
    #TODO: patch_dir.values[0].size ?
    colorMask= initiate_color_masks_gpu(nb_clust, size=ptch_size)
    cpt=0
    for k, ptc in patch_dir.items():
        cpt+=1
        patch = np.asarray(Image.open(PTCH_DIR + k + ".png"))
        patch = patch *0.7 + colorMask[ptc.colour]* 0.3
        go_faster(img, patch, ptc.get_pos[0], ptc.get_pos[1])


    px_legend = 10
    for i, col in enumerate(colorMask):
        for j in range(px_legend):
            img.paste(col, (j*ptch_size, i*ptch_size))


    print(cpt)
    return img


