from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


data = np.load("patchesArray.npy", allow_pickle=True)

nbclust=12




raw_data = np.asarray(list(data[:, 1]))
labels = np.asarray(list(data[:, 0]))

print(raw_data.shape)
nsamples, nx, ny,nz = raw_data.shape
dat_to_cluster= raw_data.reshape((nsamples,nx*ny*nz))

a = KMeans(n_clusters=nbclust).fit_predict(dat_to_cluster)

res=[ [] for _ in range(nbclust)]
for l, grp in enumerate(a):

    res[grp].append(labels[l])





fig = plt.figure()
for j in range(nbclust):
    for indx, name in enumerate(res[j][:10]):
        ax = fig.add_subplot(nbclust, 10 , indx + j*10 +1)
        img = mpimg.imread( './patches2/' + name)
        plt.imshow(img)
        plt.axis("off")
plt.show()