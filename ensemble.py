
from math import ceil
import csv
import numpy as np
from patches import Patch
from data_manipulation import process_data, flatten_raw_data
import utils.path as path
import clustering

def ensemble_clustering(image_size, patch_size, listLod, listClusters, listMethods, seuil):
    pix_clust =[ [[] for _ in range(image_size[0]//(patch_size*min(listLod)))] for __ in range(image_size[1]//(patch_size*min(listLod))) ]
    dct_lower_res = None
    lg = zip(listLod, listClusters, listMethods)
    print(path.IMG_NAME)
    for LOD, clust, method in lg:
        print(LOD, clust)
        IMG_DIR = './../Images/Patches/' + path.IMG_NAME + '_' + str(path.ptch_size) \
                  + "_LOD=" + str(LOD) + '/' + path.IMG_NAME + '_tiles/'

        dct_patch ={}
        with open(IMG_DIR + "../tile_selection2.tsv") as f:
            rcsv = csv.reader(f, delimiter="\t")
            # read the first line that holds column labels
            csv_labels = rcsv.__next__()
            for record in rcsv:
                if record[3] == '1':
                    dct_patch[record[0]] = Patch(record[0], size=path.ptch_size, row=int(record[4]), column=int(record[5]))
        if dct_lower_res is None and LOD == min(listLod):
            dimg = IMG_DIR
            dct_lower_res = dct_patch.copy()
        adda = 't' if LOD == 8 else ''
        data = np.load(IMG_DIR + "../res"+ adda + ".npy", allow_pickle=True)
        raw_data, labels = process_data(data)
        raw_data = flatten_raw_data(raw_data)
        labels = labels


        data = dict(zip(labels, raw_data))
        a, _ = method(raw_data, clust)

        dct_col = dict(zip(labels, a))
        maxi = 0

        for l, c in dct_col.items():

            ptc = dct_patch[l[:-4]]
            xo = (ceil(ptc.get_pos()[0]) // patch_size * (LOD // min(listLod)))
            yo = (ceil(ptc.get_pos()[1]) // patch_size * (LOD // min(listLod)))
            for i in range(LOD//min(listLod)):
                for j in range(LOD//min(listLod)):
                    pix_clust[yo +i][xo +j].append(c)
                    if len(pix_clust[yo][xo]) > maxi:
                        maxi= len(pix_clust[yo+i][xo+j])


    res = {}
    for l in pix_clust:
        for c in l:
            if len(c) == maxi:
                if tuple(c) not in res:
                    res[tuple(c)] = 0
                res[tuple(c)]+=1

    LOD = min(listLod)
    image_size = [ceil(image_size[0] // LOD), ceil(image_size[1] // LOD)]

    dct_col, color = ultimate_method_from_this_bogoss_wemmert(res, len(listLod), listClusters, seuil)
    print(dct_col)
    for p in dct_lower_res.values():
        col = pix_clust[p.get_pos()[1]// patch_size][p.get_pos()[0]// patch_size]

        if tuple(col) in  dct_col:
            p.colour = dct_col[tuple(col)]
        else:
            p.colour = -1

    return dct_lower_res, dimg, color,image_size


def ultimate_method_from_this_bogoss_wemmert(dClasses, nmethods, nclasses ,th, limit=8):
    print(dClasses)
    if nmethods == 1:
        res ={}
        for i in dClasses:
            res[i]=i[0]
        return res, len(res)

    transition = [[[0] * nclasses[i] for _ in range(nmethods)] for i in range(nmethods)]
    stats = {}
    best = {}
    for i in range(nmethods):
        for j in range(nmethods):
            tab_cpt = [[0] * nclasses[j] for _ in range(nclasses[i])]
            if i!=j:
                for nom, cpt in dClasses.items():
                    tab_cpt[nom[i]][nom[j]] += cpt
                for k in range(nclasses[i]):
                    transition[i][j][k] = tab_cpt[k].index(max(tab_cpt[k]))

    for  nom , cpt in dClasses.items():
        stats[nom]= 0
        vote = [[0] * nclasses[j] for j in range(nmethods)]
        for i in range(nmethods):
            k = nom[i]
            vote[i][k]+=cpt
            for j in range(nmethods):
               if j != i:
                    km = transition[i][j][k]
                    kj = nom[j]
                    vote[j][km]+=cpt
                    if kj == km:
                        stats[nom]+=cpt
        maxi=(0,0)
        vmax=0
        for i in range(nmethods):
            for j in range(nclasses[i]):
              if vmax<vote[i][j]:
                   vmax = vote[i][j]
                   maxi=(i,j)
        best[nom]= maxi
    fr={}
    corr=[]
    for  nom , cpt in dClasses.items():
        stats[nom]= stats[nom]/nmethods
        if stats[nom] >= nmethods*(1- th):
            if best[nom][0]+best[nom][1]*nmethods not in corr:
                corr.append(best[nom][0]+best[nom][1]*nmethods)

            fr[nom] = corr.index(best[nom][0]+best[nom][1]*nmethods)


    for i in dClasses:
        if i not in fr and len(fr)<limit:
            print(len(fr))
            fr[i]= len(corr)
            corr.append(max(corr)+1)

    return fr, len(corr)