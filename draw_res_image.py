from PIL import Image
from utils.path import IMG_DIR, PTCH_DIR
import csv
import numpy as np
import clustering

def initiate_color_masks(nb_clust, size):
    colorMask=[Image.new('RGB', size=size, color="hsl("+str(round(i/nb_clust*360))+ ",100%, 50%)") for i in range(nb_clust)]
    return colorMask

def draw_res_image(img_size, table, data, ptch_size, nb_clust=10, thumbsize= (15000,15000)):
    img = Image.new('RGB', size=(img_size[0], img_size[1]), color=(255,255,255))
    id = table[0]
    table = table[1:]

    raw_data = np.asarray(list(data.item().values()))
    labels = list(data.item().keys())
    print(len(labels))

    nsamples, nx, ny, nz = raw_data.shape
    dat_to_cluster = raw_data.reshape((nsamples, nx * ny * nz))
    print("start cluster")


    print(len(labels))
    a = clustering.make_clustes(dat_to_cluster, nb_clust)
    print("a", len(a))
    dct = dict(zip(labels, a))

    print("end cluster")
    colorMask= initiate_color_masks(nb_clust, size=ptch_size)
    cpt=0
    for ind, info in enumerate(table):
        if info[3]=='1':
                    cpt+=1
                    patch = Image.open(PTCH_DIR +info[0]+".png")
                    patch=Image.blend(patch, colorMask[dct[info[0]+'.png']],0.25)
                    img.paste(patch, (int(info[5])*64 , int(info[4])*64))
                    pass
    print(cpt)
    img.thumbnail(thumbsize)
    img.save("lola.jpg", "JPEG", )
