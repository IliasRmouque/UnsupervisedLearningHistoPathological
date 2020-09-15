import csv
import numpy as np
import clustering

from PIL import Image
from utils.path import IMG_DIR, PTCH_DIR
from draw_res_image import draw_res_image, draw_origin_image
from patches import Patch
from matplotlib.image import imsave


if __name__ == '__main__':
    size=(80640, 55040)
    nb_clust = 7
    ptch_size =64

    dct_patch = {}
    with open(PTCH_DIR + "../tile_selection.tsv") as f:
        rcsv = csv.reader(f, delimiter="\t")

        # read the first line that holds column labels
        csv_labels = rcsv.__next__()

        for record in rcsv:
            if record[3] == '1':
                dct_patch[record[0]]= Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))





    data = np.load("madness.npy", allow_pickle=True)
    raw_data = np.asarray(list(data.item().values()))
    labels = list(data.item().keys())

    print(len(labels))

    nsamples, nx, ny, nz = raw_data.shape
    dat_to_cluster = raw_data.reshape((nsamples, nx * ny * nz))

    print("start cluster")

    a = clustering.make_clustes(dat_to_cluster, nb_clust)

    dct_col = dict(zip(labels, a))

    print("end cluster")

    for key, colour in dct_col.items():
        #TODO: remove the try catch here bc of bad data
        try:
         dct_patch[key[:-4]].colour = colour
        except:
            pass

    print("drawing the image")

    img = draw_res_image(size, patch_dir=dct_patch, nb_clust=nb_clust)


    print("saving the thumbnail")

    img.thumbnail((10000, 10000))
    img.save("Original.jpg", "jpeg")
    img.close()  # suuuuuuuper important