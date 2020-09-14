import imageio
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.path import IMG_DIR, PTCH_DIR
import png

data = np.load("autoencoded_pictures.npy", allow_pickle=True)
nbclust=12

def make_clustes(dat_to_cluster , nbclust=4):

    a = KMeans(n_clusters=nbclust).fit_predict(dat_to_cluster)
    return a

if __name__ == '__main__':

    raw_data = np.asarray(list(data.item().values()))
    labels = list(data.item().keys())

    print(raw_data.shape)
    nsamples, nx, ny, nz = raw_data.shape
    dat_to_cluster = raw_data.reshape((nsamples, nx * ny * nz))
    res=[ [] for _ in range(nbclust)]






    fig = plt.figure()
    for j in range(nbclust):
        for indx, name in enumerate(res[j][:10]):
            ax = fig.add_subplot(nbclust, 10 , indx + j*10 +1)
            img = mpimg.imread(PTCH_DIR + name)
            plt.imshow(img)
            plt.axis("off")
    plt.show()
    img = imageio.imread(IMG_DIR + name)

    np.zeros()
