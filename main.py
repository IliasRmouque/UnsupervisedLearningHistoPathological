from PIL import Image
from utils.path import IMG_DIR, PTCH_DIR
from draw_res_image import draw_res_image
import csv
import numpy as np
import clustering



if __name__ == '__main__':
    size=(80640, 55040)
    nb_clust = 10

    table = csv.DictReader(open(PTCH_DIR + "../tile_selection.tsv"), delimiter="\t")
    data = np.load("madness.npy", allow_pickle=True)
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

    draw_res_image(size, table, clust)

