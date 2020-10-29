from random import sample

import imageio
from cuml.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.path import IMG_DIR, PTCH_DIR
import faulthandler; faulthandler.enable()

data = np.load("autoencoded_pictures.npy", allow_pickle=True)

def mean_color_pixel_association(data):
    dat_to_cluster = np.asarray(list(data.item().values())).mean(1).mean(1) / 255
    labels = list(data.item().keys())
    return dat_to_cluster, labels

def make_clusters(dat_to_cluster , nb_clust=4):
    estimator = KMeans(n_clusters=nb_clust)
    res = estimator.fit_predict(dat_to_cluster)
    return res, estimator


def make_clusters_2(dat_to_cluster , nb_clust, n=15000):
    estimator = KMeans(n_clusters=nb_clust)
    if n < len(dat_to_cluster):
        s=np.asarray(sample(list(dat_to_cluster), k = n), dtype=np.float_)
        estimator= estimator.fit(s)
        res = []
        notinit = True
        for i in range(0, len(dat_to_cluster) // n + 1):
            d = np.asarray(list(dat_to_cluster)[n * i: n * (i + 1)])
            if notinit:
                res = list(estimator.predict(d))
                notinit = False
            else:
                a =list(estimator.predict(d))

                for j in a:
                    res.append(j)


    else :
        res = estimator.fit_predict(dat_to_cluster)
    return res, estimator


if __name__ == '__main__':
    pass
    # nbclust=4
    # data = (np.load("../Images/Test2/patchesArray2.npy", allow_pickle=True))
    #
    #
    # raw_data = np.asarray(list(data.item().values())[:10000]).mean(1).mean(1)/255
    # labels = list(data.item().keys())
    #
    # print(raw_data.shape)
    # """
    # nsamples, nx, ny, nz = raw_data.shape
    # dat_to_cluster = raw_data.reshape((nsamples, nx * ny * nz))
    # """
    # a = make_clusters(raw_data , nbclust)
    #
    # res = [[] for _ in range(nbclust)]
    #
    # for i in range(len(a)):
    #     res[a[i]].append(labels[i])
    #
    # fig = plt.figure()
    # for j in range(nbclust):
    #     for indx, name in enumerate(res[j][:10]):
    #         ax = fig.add_subplot(nbclust, 10 , indx + j*10 +1)
    #         img = mpimg.imread(IMG_DIR + name )
    #         plt.imshow(img)
    #         plt.axis("off")
    # plt.show()


