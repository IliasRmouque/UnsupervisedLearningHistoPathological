from random import sample

import imageio
from cuml.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, SpectralClustering,MeanShift

import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

import matplotlib.image as mpimg
from utils.path import IMG_DIR, PTCH_DIR
import faulthandler; faulthandler.enable()



def mean_color_pixel_association(data):
    dat_to_cluster = np.asarray(list(data.item().values())).mean(1).mean(1) / 255
    labels = list(data.item().keys())
    return dat_to_cluster, labels

def make_clusters(dat_to_cluster , nb_clust=4):
    estimator = KMeans(n_clusters=nb_clust)
    res = estimator.fit_predict(dat_to_cluster)
    return res, estimator


def make_clusters_KMeans(dat_to_cluster , nb_clust, n=15000):
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

#### DBSANC ZONE

def find_eps(dataset):

    nearest_neighbors = NearestNeighbors(n_neighbors=6)
    neighbors = nearest_neighbors.fit(dataset)
    distances, indices = neighbors.kneighbors(dataset)
    distances = np.sort(distances[:,5], axis=0)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel("Distance")
    # plt.show()
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    fig = plt.figure(figsize=(5, 5))
    knee.plot_knee()
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.savefig("Distance_curve.png", dpi=300)
    return distances[knee.knee]

def make_clusters_DBSCAN(dat_to_cluster, eps):
    estimator = DBSCAN(eps= eps, min_samples=3)
    res = estimator.fit_predict(dat_to_cluster)
    return res, estimator

def make_clusters_DBSCAN(dat_to_cluster, eps):
    estimator = DBSCAN(eps= eps, min_samples=3)
    res = estimator.fit_predict(dat_to_cluster)
    return res, estimator

def make_clusters_Mean(dat_to_cluster):
    estimator = MeanShift()
    res = estimator.fit_predict(dat_to_cluster)
    return res, estimator

def make_clusters_SpectralClustering(dat_to_cluster, n):
    estimator = SpectralClustering(n_clusters=n, n_jobs=-1)
    res = estimator.fit_predict(dat_to_cluster)
    return res, estimator


def make_clusters_Gaussian(dat_to_cluster, n=15000):
    estimator = GaussianMixture(n_components=4)
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


def plot_dendrogram(dataset):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(dataset)

    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

def make_clusters_Agglo(dat_to_cluster, n):
    estimator = AgglomerativeClustering(n_clusters=n, linkage="ward")
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


