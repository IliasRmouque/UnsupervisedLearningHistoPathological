import csv
import tempfile
from math import ceil, floor

from random import sample
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
from shapely import wkt
from matplotlib import pyplot as plt
from descartes.patch import PolygonPatch
from utils.path import *
from draw_res_image import draw_res_image, draw_origin_image
from patches import Patch
from matplotlib.colors import hsv_to_rgb

import umap
import imageio
import numpy as np
from yellowbrick.cluster import kelbow_visualizer
import matplotlib.patches as mpatches
import clustering
import shapely
import seaborn as sns
from os import listdir, mkdir, makedirs
from PIL import Image
from matplotlib.cbook import get_sample_data

def addfigure_xy(x, y, ax, file_dir, zoom=1.0, imgformat='png'):
    """
    Add a figure (from file direction) to a [x, y] position to a plot (stored in ax)
    :param x: Coordenate x to append file
    :param y: Coordenate y to append file
    :param ax: Matlab plot object to append file
    :param file_dir: Direction to file
    :param zoom: Image zoom
    :param imgformat: Format of the image
    :return:
    """

    # Load image
    arr_img = plt.imread(file_dir, format=imgformat)
    imagebox = OffsetImage(arr_img, zoom=zoom)
    imagebox.image.axes = ax

    # Create Annotation box
    ab = AnnotationBbox(imagebox, [x, y], frameon=False)
    ax.add_artist(ab)

def show_clustering(dct, nb_clust, IMG_DIR, nbr=10 ):
    a=[[] for _ in range(nb_clust)]
    for l, c in dct.items():
        if len(a[c-1])<10:
            a[c-1].append(l)
    fig = plt.figure()
    for j in range(nb_clust):
        for i in range(nbr):
            if i>=len(a[j]):
                break
            ax = fig.add_subplot(nb_clust, nbr, i + 1 + j*nbr)
            img = imageio.imread(IMG_DIR + a[j][i])
            plt.imshow(img)
            plt.axis("off")
    plt.show()

def draw_square_xy(anno, ax, nb = 10):
    a=[]
    nbsquare = 0
    max_col = len(anno)
    for i in anno:
        if i:
            a.append((i[0], len(i)))
            nbsquare += len(i)



    boxes= []
    cpt = 0
    ind = 0
    if a:
        for j in range(nb**2):

            if cpt > ceil(a[ind][1]*nb**2 / nbsquare +1):
                cpt=0
                ind+=1
                while not a[ind]:
                    ind+=1
            sh = shapely.geometry.Polygon(
                [(j % nb , j // nb ),
                    (j % nb + 1, j // nb ),
                    (j % nb + 1, j // nb + 1),
                    (j % nb, j // nb + 1 )])
            ax.add_patch(PolygonPatch(sh, color=hsv_to_rgb([a[ind][0]/len(anno), 0.7, 1])))
            cpt += 1



def confusion_matrix(pred, real,  nb_classes, nb_clust,  ax, classes):
    ax.set_aspect("equal")
    hist, xbins, ybins, im = ax.hist2d(pred, real, bins=(nb_clust, nb_classes), cmap=plt.cm.winter)
    ax.xlabel = "pred"
    ax.ylabel = "real"
    if classes is not None:
        ax.yaxis.set_ticklabels(classes)

    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            ax.text(xbins[j] + xbins[-1]/nb_clust/2, ybins[i] + ybins[-1]/nb_classes/2, hist.T[i, j],
                    color="w", ha="center", va="center", fontweight="bold")

def clustering_and_img_drawin(image_size, dct_patch, IMG_DIR, nb_clust = 12, res_name= "res", thumb_size=(7000, 7000)):

    print("drawing the image")
    img = draw_res_image(image_size, patch_dic=dct_patch,folder=IMG_DIR, nb_clust=nb_clust, downscale_factor=4)
    print("saving the thumbnail")
    img.thumbnail(thumb_size)
    img.save(res_name + ".jpg", "jpeg")
    img.close()  # suuuuuuuper important


def ultimate_method_from_this_bogoss_wemmert(dClasses, nmethods, nclasses ,th=0.5):
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
    return fr, len(corr)


def ensemble_clustering(image_size, patch_size, listLod, listClusters,  seuil=0.6, terminaison='', namefolder='./',suffixe=''  ):
    pix_clust =[ [[] for _ in range(image_size[0]//(patch_size*listLod[0]))] for __ in range(image_size[1]//(patch_size*listLod[0])) ]
    dct_lower_res = None
    lg = zip(listLod, listClusters)
    for LOD, clust in lg:
     for ds in[0,1]:
        print(LOD, clust)
        IMG_DIR = './../PyHIST/output/' + IMG_NAME + '_' + str(ptch_size) + '*' + str(ptch_size) \
              + "_LOD=" + str(LOD) + '/' + IMG_NAME + '_tiles/'
        dct_patch ={}
        with open(IMG_DIR + "../tile_selection2.tsv") as f:
            rcsv = csv.reader(f, delimiter="\t")
            # read the first line that holds column labels
            csv_labels = rcsv.__next__()
            for record in rcsv:
                if record[3] == '1':
                    dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))
        if dct_lower_res is None:
            dimg = IMG_DIR
            dct_lower_res = dct_patch.copy()
        data = np.load(IMG_DIR + "../pa"+str(ds)+".npy", allow_pickle=True)
        raw_data, labels = process_data(data)
        raw_data = flatten_raw_data(raw_data)
        labels = labels


        data = dict(zip(labels, raw_data))
        a, _ = clustering.make_clusters_2(raw_data, clust, 110000)

        dct_col = dict(zip(labels, a))
        maxi = 0

        for l, c in dct_col.items():

            ptc = dct_patch[l[:-4]]
            xo = (ceil(ptc.get_pos()[0]) // patch_size * (LOD // listLod[0]))
            yo = (ceil(ptc.get_pos()[1]) // patch_size * (LOD // listLod[0]))
            for i in range(LOD//listLod[0]):
                for j in range(LOD//listLod[0]):
                    pix_clust[yo +i][xo +j].append(c)
            if len(pix_clust[yo][xo]) > maxi:
                maxi= len(pix_clust[yo][xo])


    res = {}
    for l in pix_clust:
        for c in l:
            if len(c) == maxi:
                if tuple(c) not in res:
                    res[tuple(c)] = 0
                res[tuple(c)]+=1

    LOD = listLod[0]
    image_size = [ceil(image_size[0] // LOD), ceil(image_size[1] // LOD)]
    # clustering_and_img_drawin(image_size, dct_lower_res, dimg, color, res_name=namefolder + "dessin",
    #                           thumb_size=(2500, 2500))

    # def ptree(start, tree, indent_width=4, ):
    #
    #     def _ptree(start, parent, tree, grandpa=None, indent=""):
    #         if parent != start:
    #             if grandpa is None:  # Ask grandpa kids!
    #                 print(parent,':',res[parent], end="")
    #             else:
    #                 print(parent,':',res[parent])
    #         if parent not in tree:
    #             return
    #         for child in tree[parent][:-1]:
    #             print(indent + "├" + "─" * indent_width, end="")
    #             _ptree(start, child, tree, parent, indent + "│" + " " * 4)
    #         child = tree[parent][-1]
    #         print(indent + "└" + "─" * indent_width, end="")
    #         _ptree(start, child, tree, parent, indent + " " * 5)  # 4 -> 5
    #
    #     parent = start
    #     _ptree(start, parent, tree)
    #
    #
    # tmp = res.copy()
    # tree = {tuple(): []}
    # for i in range(maxi+1):
    #     for c, nbr in tmp.items():
    #         if c[0:i] not in res:
    #             res[c[0:i]]=0
    #         if i < maxi:
    #             res[c[0:i]] += res[c]
    #         # if i == 0 :
    #         #     tree[c[0:i]]=[]
    #         #     if c[0:i] not in tree[tupl]:
    #         #         tree[-1].append(c[0:i])
    #         # else:
    #         if i!=0:
    #             if c[0:i - 1] not in tree:
    #                 tree[c[0:i - 1]] = []
    #             if c[0:i] not in tree[c[0:i-1]]:
    #                 tree[c[0:i-1]].append(c[0:i])
    #
    #
    #
    # ptree(tuple(), tree)
    #
    # def rmClass(start, tree, val ):
    #     def removeChilds(parent, tree):
    #         for child in tree[parent]:
    #             if child in tree:
    #                 removeChilds(child, tree)
    #         tree.pop(parent)
    #
    #
    #
    #     def clear(start, parent, tree, val, go=False, indent=""):
    #         if go:
    #             lchild = []
    #             al = None
    #             for child in tree[parent]:
    #                 if val[child]/val[parent] > seuil:
    #                     al = child
    #                     break
    #             if al is not None:
    #                 for child in tree[parent][:]:
    #                     if child in tree:
    #                         removeChilds(child, tree)
    #             else:
    #                 for child in tree[parent][:]:
    #                     if child in tree:
    #                         clear(child, child, tree, val, True)
    #         else:
    #             for child in tree[parent][:]:
    #                 clear(parent, child, tree, val, True)
    #
    #     parent = start
    #     clear(start, parent, tree, val, True)
    #
    # print(len(tree))
    # #ptree(tuple(), tree)
    # rmClass(tuple(), tree ,res)
    # ptree(tuple(), tree)
    # #print(len(tree))
    #
    # leaves =[]
    # for k, i in tree.items():
    #     if i[0] not in tree:
    #         leaves.append(k)
    #
    # # def grand_maraboutage_recursif(liste, chose, j, maxi):
    # #     if len(chose)<maxi:
    # #         if j<maxi:
    # #             c = chose[:]
    # #             grand_maraboutage_recursif(liste, c, j + 1, maxi)
    # #             chose.append(j)
    # #             grand_maraboutage_recursif(liste, chose, 0, maxi)
    # #
    # #     else :
    # #         if chose not in liste:
    # #             liste.append(chose)
    # #
    # #
    #
    # # color =-1
    # # dct_col = {}
    # # for l in leaves:
    # #     color+=1
    # #     pc = []
    # #     grand_maraboutage_recursif(pc, list(l), 0, len(listLod))
    # #     for c in pc:
    # #         dct_col[tuple(c)]=color
    # print(leaves)
    # color = 0
    # dct_col = {}
    # for l in leaves:
    #     dct_col[tuple(l)] = color
    #     color += 1
    # corr={}
    # for nom in res.keys():
    #     corr[nom]=nom
    # Pnom = list(res.keys())
    # for st in range(len(Pnom)):
    #     for ne in range(len(Pnom)-st-1):
    #         current=Pnom[st]
    #         next=Pnom[st+ne+1]
    #         cpt=0
    #         for i in range(len(current)):
    #             if current[i] == next[i]:
    #                 cpt+=1
    #         if cpt>= th *len(listLod):
    #             corr[current]=next
    #             break
    #
    # dct_col={}
    # color=0
    # for nom, c in corr.items():
    #     if nom == c:
    #         dct_col[nom]=color
    #         color+=1
    # for nom, c in corr.items():
    #     cur= c
    #     while corr[cur] != cur :
    #         cur = corr[cur]
    #     dct_col[nom]=dct_col[cur]









    #  "Salut"
    # # for i  in range(len(pix_clust)):
    # #     for j  in range(len(pix_clust[i])):
    # #         if len(pix_clust[i][j]) == maxi:
    # #             pix_clust[i][j] = hsv_to_rgb([dct_col[tuple(pix_clust[i][j])] / color, 0.5, 1])
    # #         else:
    # #             pix_clust[i][j] = hsv_to_rgb([0, 0, 0])
    # # print(np.asarray(pix_clust).shape)
    # # img = Image.fromarray(np.asarray(pix_clust)*255, 'RGB')
    # # img.save('./res.png')
    # # plt.imshow(np.asarray(pix_clust), interpolation='nearest')
    # # plt.show()
    #

    dct_col, color = ultimate_method_from_this_bogoss_wemmert(res, len(listLod), listClusters, th)
    print(dct_col)
    for p in dct_lower_res.values():
        col = pix_clust[p.get_pos()[1]// patch_size][p.get_pos()[0]// patch_size]

        if tuple(col) in  dct_col:
            p.colour = dct_col[tuple(col)]
        else:
            p.colour = -1


    clustering_and_img_drawin(image_size, dct_lower_res, dimg, color, res_name=namefolder+"dessin", thumb_size=(2500,2500))
    mask_type = load_annotations("./annoWeird.csv", "./anno.csv", IMG_ID, color)
    data_visualisation(color, mask_type, dct_lower_res, image_size, LOD, IMG_ID, terminaison=terminaison, namefolder=namefolder,suffixe=suffixe  )









#def clustering_and_img_drawin(data, image_size, nb_clust = 12, ptch_size = 64, res_name= "res"):

def data_visualisation( nb_clust, mask_type, dct_patch, IMG_SIZE, LOD, img_id, terminaison='', namefolder='./',suffixe='', nClassest=2 ):


    # print('Total time in seconds:', interval)

    term_score = {}
    class_color = 0
    fig, ax = plt.subplots()

    mask_type = load_annotations("./annoWeird.csv", "./anno.csv", img_id, nb_clust, ax)

    pred = []
    real = []
    classesorder = []
    for key, ptch in dct_patch.items():

        try:
            colour = ptch.colour
            if colour != -1 :
                sh = shapely.geometry.Polygon(
                    [(ptch.column * ptch.size * LOD, IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD),
                     (ptch.column * ptch.size * LOD + ptch.size * LOD, IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD),
                     (ptch.column * ptch.size * LOD + ptch.size * LOD,
                      IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD + ptch.size * LOD),
                     (ptch.column * ptch.size * LOD, IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD + ptch.size * LOD)])
                for i, j in mask_type.items():
                    if j["WKT"].contains(sh):  # j["WKT"].intersects(sh): #
                        if j["Term"] not in term_score:
                            term_score[j["Term"]] = {}
                            term_score[j["Term"]]["Predicted"] = []
                            term_score[j["Term"]]["Patches"] = []
                            term_score[j["Term"]]["Real"] = class_color
                            classesorder.append(j["Term"])
                            class_color += 1
                        ax.add_patch(PolygonPatch(sh, color=hsv_to_rgb( [colour/ nb_clust, 1, 1])))
                        j["Clust"][colour].append(colour)
                        term_score[j["Term"]]["Predicted"].append(colour)
                        term_score[j["Term"]]["Patches"].append(key)
                        pred.append(colour)
                        real.append(term_score[j["Term"]]["Real"])
        except KeyError:
            pass
    print("Pour", nb_clust, "clusters \nARI=", se, "\nNMI=",
          normalized_mutual_info_score(real, pred))
    predt=[]
    rt=[]
    for i in range(len(pred)):
        if real[i] == term_score["tumor"]["Real"]:
            predt.append(pred[i])
            rt.append(term_score["tumor"]["Real"])
    #True positive False_Positive for tumor

    maxs=[0]*nClassest
    t_colors=[-1]*nClassest
    for i in range(nb_clust):
        cpt = predt.count(i)
        mem=i
        for m in range(nClassest):
            if maxs[m]<cpt:
                b = t_colors[m]
                t_colors[m] = mem
                mem = b
                a = maxs[m]
                maxs[m] = cpt
                cpt = a
    cptf=0
    cptt=0
    for i in range(nClassest):
        cptt += predt.count(t_colors[i])
        cptf += pred.count(t_colors[i])
    print(t_colors)
    print(cptt)
    print(len(predt))
    vp = cptt
    fp = cptf - cptt
    vn = len(pred) - len(predt) - cptf + cptt
    fn = len(predt) - cptt

    print("vrai positif:", vp)
    print("faux positif:", fp )
    print("vrai négatif:", vn )
    print("faux négatif:",fn)
    print("sensibilité:", vp / (vp + fn))
    print("spécificité:", vn / (vn + fp))





    classesorder ={}
    for k, i in term_score.items():
        classesorder[len(i["Predicted"])] = k
        print(k, len(i["Predicted"]))


    #just show the
    fig.set_size_inches(30, 15)
    plt.title("coloration des zones déjà annotées" + terminaison)
    plt.savefig(namefolder + "coloration_zones_annotées_" + terminaison + suffixe+ ".png", format="png")
    # ax.cla()

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    confusion_matrix( pred, real, class_color, nb_clust, ax, list(classesorder.values()))

    plt.title("Matrice de confusion" + terminaison)
    plt.savefig(namefolder + "Matrice_confusion_" + terminaison +suffixe+ ".png", format="png")
    ax.cla()

    nb_square = 100
    a = [(j["Term"], j) for i, j in mask_type.items()]

    rms = []
    for i, j in a:
        rm = True
        for b in j["Clust"]:
            if b:
                rm = False
                break
        if rm:
            rms.append((i, j))

    for r in rms:
        a.remove(r)
    a = sample(a, k=min(nb_square, len(a)))
    a = sorted(a, key=lambda x: x[0])

    nb_lines = 5

    x = 1
    fig, ax = plt.subplots()
    for i, j in a:
        ax = plt.subplot(nb_lines, nb_square // nb_lines, x)
        ax.set_title(j["Term"])
        draw_square_xy(j["Clust"], ax)
        ax.axis([0, 10, 0, 10])
        ax.axis("off")
        x += 1
    fig.set_size_inches(30, 15)
    plt.savefig(namefolder + "coloration_carrés_annotées" + terminaison + suffixe+ ".png", format="png")
    plt.close('all')
    return term_score


def process_data(data):
    raw_data = np.asarray(list(data.item().values()), dtype=np.float_)
    labels = list(data.item().keys())
    return raw_data, labels

def flatten_raw_data(raw_data):
    return raw_data.reshape((raw_data.shape[0], np.prod(raw_data.shape[1:])))


def load_annotations(file_with_Term, file_with_Polygon, img_id, nb_clust, ax=None):
    mask_type = {}
    with open(file_with_Term, mode='r') as f:
        rcsv = csv.DictReader(f, delimiter="\t")
        line_count = 0
        for record in rcsv:
            if record["ImageId"] == img_id:
                mask_type[record["Id"]] = {}
                mask_type[record["Id"]]["Clust"] = [[] for i in range(nb_clust)]
                mask_type[record["Id"]]["Term"] = record["Term"]
                mask_type[record["Id"]]["Patches"] = []

    with open(file_with_Polygon, mode='r') as f:
        rcsv = csv.DictReader(f, delimiter=";")
        for record in rcsv:
            try:
                if record["Image"] == img_id:
                    mask_type[record["ID"]]["WKT"] = wkt.loads(record["WKT "])
                    if ax is not None:
                        ax.plot(*wkt.loads(record["WKT "]).exterior.coords.xy)
            except KeyError:
                pass
    return mask_type

def get_termlist(dct_patch,mask_type, nb_clust, IMG_SIZE, LOD):
    term_list = {}
    class_color = 0
    for key, ptch in dct_patch.items():
            colour = ptch.colour
            # if colour != -1:
            sh = shapely.geometry.Polygon(
                [(ptch.column * ptch.size * LOD, IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD),
                 (ptch.column * ptch.size * LOD + ptch.size * LOD,
                  IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD),
                 (ptch.column * ptch.size * LOD + ptch.size * LOD,
                  IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD + ptch.size * LOD),
                 (ptch.column * ptch.size * LOD,
                  IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD + ptch.size * LOD)])
            for i, j in mask_type.items():
                if j["WKT"].contains(sh):  # j["WKT"].intersects(sh): #
                    if j["Term"] not in term_list:
                        term_list[j["Term"]] = {}
                        term_list[j["Term"]]["Predicted"] = [[] for i in range(nb_clust)]
                        term_list[j["Term"]]["Patches"] = []
                        term_list[j["Term"]]["Real"] = class_color
                        class_color += 1
                    j["Clust"][colour].append(colour)
                    term_list[j["Term"]]["Patches"].append(key)
                    ptch.term=j["Term"]
    return term_list


def get_UMAP(data, ax, nb_clust, term_list, nb_limit=500):
    clist = []
    tlist=[]
    rlist=[]
    for t, a in term_list.items():
        i = 0

        if t != "trash":
            tlist.append(t)
            for n in a["Patches"]:
                if i > nb_limit:
                    break
                rlist.append(data[n + '.png'])
                clist.append(a["Real"])
                i += 1
    reducer = umap.UMAP()
    print(len(rlist))
    scaled_data = StandardScaler().fit_transform(rlist)
    embedding = reducer.fit_transform(scaled_data)
    a = clist[0]
    classe = 0
    if ax is None:
        ax = plt.subplot(111)

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[hsv_to_rgb([ x / nb_clust, 1, 1]) for x in clist]
    )
    leg=[]
    for t in tlist:
        red_patch = mpatches.Patch(color=hsv_to_rgb([term_list[t]["Real"]/nb_clust, 1, 1 ]), label=t)
        leg.append(red_patch)
    ax.legend(handles=leg)
    ax.set_title('UMAP projection of the dataset')
    return ax

def get_UMAP_classes(data_list, dct_patch, nb_clust, mask_type, IMG_SIZE, IMG_DIR):
    shift=0
    fig, axes = plt.subplots(7, len(data_list), sharex=True, sharey=True)
    axes = axes.flatten()
    for data_name in data_list:
        print("UMAP ", data_name)
        term_list = get_termlist(dct_patch, mask_type, nb_clust, IMG_SIZE, LOD)
        data = np.load(IMG_DIR + data_name, allow_pickle=True)
        raw_data, labels = process_data(data)
        raw_data = flatten_raw_data(raw_data)
        data = dict(zip(labels, raw_data))
        UMAP_for_a_class(data, nb_clust, term_list, axes, len(data_list), shift, nb_limit=1000)
        shift+=1
    plt.show()




def UMAP_for_a_class(data, nb_clust, term_list, axes,scale, shift, nb_limit=500):
            clist = []
            tlist = []
            rlist = []
            import umap
            for t, a in term_list.items():
                i = 0
                if t != "trash":
                    tlist.append(t)
                    for n in a["Patches"]:
                        if i > nb_limit:
                            break
                        rlist.append(data[n + '.png'])
                        clist.append(a["Real"])
                        i += 1
            reducer = umap.UMAP()

            scaled_penguin_data = StandardScaler().fit_transform(rlist)
            embedding = reducer.fit_transform(scaled_penguin_data)

            a = clist[0]
            classe = 0
            if axes is None:
                fig, axes = plt.subplots(len(tlist), 1, sharex=True, sharey=True,subplot_kw={'aspect': 0.4, 'adjustable': 'box'})
                axes = axes.flatten()
            ax = axes[scale * classe + shift]
            ax.set_title(tlist[classe])
            plist = []
            for i, p in enumerate(embedding):
                if clist[i] != a:
                    plist = np.asarray(plist)
                    ax.scatter(plist[:, 0],
                               plist[:, 1],
                               color=[hsv_to_rgb([clist[i-1] / nb_clust, 1, 1]) for _ in range(len(plist))],
                               alpha=0.7)
                    classe += 1
                    ax = axes[scale * classe +shift]
                    ax.set_title(tlist[classe])
                    plist=[]
                plist.append(p)
                a = clist[i]
            plist = np.asarray(plist)
            ax.scatter(plist[:, 0], plist[:, 1],
                       color=[hsv_to_rgb([clist[i] / nb_clust, 1, 1]) for _ in range(len(plist))], alpha=0.7)




if __name__ == '__main__':
    #
    # print("Normal")
    # namefolder = "./Normal2/"
    # makedirs(namefolder, exist_ok=True)
    # dct_patch = {}
    # nb_clust =4
    # mask_type = load_annotations("./annoWeird.csv", "./anno.csv", IMG_ID, nb_clust)
    # with open(IMG_DIR + "../tile_selection2.tsv") as f:
    #     rcsv = csv.reader(f, delimiter="\t")
    #     # read the first line that holds column labels
    #     csv_labels = rcsv.__next__()
    #     for record in rcsv:
    #         if record[3] == '1':
    #             dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))
    #
    # data = np.load(IMG_DIR + "../predicted_array_512.npy", allow_pickle=True)
    # raw_data, labels = process_data(data)
    # raw_data = flatten_raw_data(raw_data)
    # labels = labels
    # data = dict(zip(labels, raw_data))
    # print("start cluster")
    # a, _ = clustering.make_clusters_2(raw_data, nb_clust, 110000)
    # dct_col = dict(zip(labels, a))
    # print("end cluster")
    #
    #
    # for l, c in dct_col.items():
    #     try:
    #      dct_patch[l[:-4]].colour = c
    #     except:
    #         print(l)
    #
    #
    # clustering_and_img_drawin(IMG_SIZE, dct_patch, IMG_DIR, nb_clust=nb_clust, res_name=namefolder+'res', thumb_size=(7000, 7000))
    #
    # term_list = data_visualisation(nb_clust, mask_type, dct_patch, IMG_SIZE, LOD, IMG_ID, '', namefolder)

    nbc = 10
    minC = 6
    th=0.7
    print("Ensemble2")
    nd= "./Contrôle/EnsembleMulti2/"
    makedirs(nd, exist_ok=True)

    LODs=[4]
    ll=LODs*nbc

    lc=[minC+ i // len(LODs) for i in range(nbc*len(LODs))]
    print(ll,lc)
    ensemble_clustering([80640, 55040], ptch_size, listLod=ll, listClusters=lc,seuil=th, namefolder=nd)





    # # data_list = ["../patchesArray.npy", "../predicted_array3.npy" ]
    # # nb_clust = 7
    # # dct_patch = {}
    # # with open(IMG_DIR + "../tile_selection2.tsv") as f:
    # #     rcsv = csv.reader(f, delimiter="\t")
    # #     # read the first line that holds column labels
    # #     csv_labels = rcsv.__next__()
    # #     for record in rcsv:
    # #         if record[3] == '1':
    # #             dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))
    # # mask_type = load_annotations("./annoWeird.csv", "./anno.csv", IMG_ID, nb_clust)
    # # data = np.load(IMG_DIR + "../patchesArray.npy", allow_pickle=True)
    # # raw_data, labels = process_data(data)
    # # raw_data = flatten_raw_data(raw_data)
    # # labels = labels
    # # data = dict(zip(labels, raw_data))
    # #ax =plt.subplot(111)
    # #get_UMAP(data, ax, nb_clust, get_termlist(dct_patch, mask_type, 0, IMG_SIZE, LOD), nb_limit=500)
    # #get_UMAP_classes(data_list, dct_patch, nb_clust, mask_type, IMG_SIZE, IMG_DIR)
    # #plt.show()
    # exit(0)
    #
    #
    #
    #
    # dct_patch = {}
    # with open(IMG_DIR + "../tile_selection2.tsv") as f:
    #     rcsv = csv.reader(f, delimiter="\t")
    #     # read the first line that holds column labels
    #     csv_labels = rcsv.__next__()
    #     for record in rcsv:
    #         if record[3] == '1':
    #             dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))
    #
    # data = np.load(IMG_DIR + "../patchesArray.npy", allow_pickle=True)
    # raw_data, labels = process_data(data)
    # raw_data = flatten_raw_data(raw_data)
    # labels = labels
    # data = dict(zip(labels, raw_data))
    # for zsiz in [128,2048]:
    #  for nb_clust in [4]:
    #     data = np.load(IMG_DIR + "../predicted_array_"+ str(zsiz) +".npy", allow_pickle=True)
    #     raw_data, labels = process_data(data)
    #     raw_data = flatten_raw_data(raw_data)
    #     labels = labels
    #     data = dict(zip(labels, raw_data))
    #     #fig, ax = plt.subplots()
    #     mask_type = load_annotations("./annoWeird.csv", "./anno.csv", IMG_ID, nb_clust)
    #
    #     terminaison = "N_Clust=" + str(nb_clust) + "_Ptch_Size=" + str(ptch_size) + "_LOD=" + str(LOD)
    #     namefolder= IMG_DIR+ "../Res_Images/N_Clust_" + str(zsiz) +"/"
    #     makedirs(namefolder, exist_ok=True)
    #
    #     # basic
    #     print("start cluster")
    #     a, _ = clustering.make_clusters_2(raw_data, nb_clust, 110000)
    #     dct_col = dict(zip(labels, a))
    #     print("end cluster")
    #     # show_clustering(dct_col, nb_clust=nb_clust)
    #
    #     for l, c in dct_col.items():
    #         try:
    #          dct_patch[l[:-4]].colour = c
    #         except:
    #             print(l)
    #
    #
    #     clustering_and_img_drawin(IMG_SIZE, dct_patch, IMG_DIR, nb_clust=nb_clust, res_name=namefolder+'res', thumb_size=(7000, 7000))
    #
    #     term_list = data_visualisation(nb_clust, mask_type, dct_patch, IMG_SIZE, LOD, IMG_ID, terminaison, namefolder)



"""# TODO: Change paramaters to launch main with a command line
    nb_clust = 8
    ptch_size = 64
    ax = plt.subplot(1, 1, 1)

    IMG_SIZE = [80640, 55040]


    # loading both annotation downloading on the site you have the term but no polygon and dl with the API you
    # have the polygons but no terms so there I merge both of them
    data = np.load("madness.npy", allow_pickle=True)
    raw_data, labels = process_data(data)
    dat_to_cluster = flatten_raw_data(raw_data)

    print("start cluster")
    print(data.shape)
    a, estimator  = clustering.make_clustes(dat_to_cluster, nb_clust)
    dct_col = dict(zip(labels, a))
    print("end cluster")

    pred = []
    real = []
    class_color=0
    cpt=0
    term_score= {}

    # loading patch infos
    dct_patch = {}
    with open(PTCH_DIR + "../tile_selection.tsv") as f:
        rcsv = csv.reader(f, delimiter="\t")

        # read the first line that holds column labels
        csv_labels = rcsv.__next__()
        for record in rcsv:
            if record[3] == '1':
                dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))

    mask_type =load_annotations("./annoWeird.csv", "./anno.csv")

    for key, colour in dct_col.items():
        # TODO: remove the try catch here bc of bad data
        try:
            a= dct_patch[key[:-4]]
            sh = shapely.geometry.Polygon(
                [(a.column * a.size, IMG_SIZE[1] - a.row * a.size),
                 (a.column * a.size + a.size, IMG_SIZE[1] - a.row * a.size),
                 (a.column * a.size + a.size, IMG_SIZE[1] - a.row * a.size + a.size),
                 (a.column * a.size, IMG_SIZE[1] - a.row * a.size + a.size)])
            for i, j in mask_type.items():
                if j["WKT"].contains(sh) or j["WKT"].crosses(sh):
                    if j["Term"] not in term_score:
                        term_score[j["Term"]] = {}
                        term_score[j["Term"]]["Predicted"] = [[] for i in range(nb_clust)]
                        term_score[j["Term"]]["Real"] = class_color
                        class_color += 1
                    #ax.add_patch(PolygonPatch(sh, color=hsv_to_rgb([colour/nb_clust, 1, 1])))
                    j["Clust"][colour].append(colour)
                    term_score[j["Term"]]["Predicted"][colour].append(colour)
                    pred.append(colour)
                    real.append(term_score[j["Term"]]["Real"])

                    break
        except KeyError:

            cpt += 1
            pass
    plt.show()

    nb_square  = len(mask_type.items())
    print(nb_square)
    x = 1

    a=[(j["Term"],j) for i, j in mask_type.items()]
    a= sorted(a, key=lambda x: x[0])
    rms=[]
    for i, j in a:
        rm = True
        for b in j["Clust"]:
            if b:
                rm =False
                break
        if rm:
            rms.append((i,j))
            nb_square -=1

    for r in rms:
        a.remove(r)

    for i,j in a:
            ax = plt.subplot(3, nb_square//3 +1, x)
            ax.set_title(j["Term"])
            draw_square_xy(j["Clust"], ax)
            ax.axis([0, 10, 0, 10])
            ax.axis("off")
            x+=1
    plt.show()"""


"""# TODO: Change paramaters to launch main with a command line
nb_clust = 4
ptch_size = 64
ax = plt.subplot(1, 1, 1)

IMG_SIZE = [80640, 55040]

# loading both annotation downloading on the site you have the term but no polygon and dl with the API you
# have the polygons but no terms so there I merge both of them
# data = np.load("madness.npy", allow_pickle=True)

pred = []
real = []

class_color = 0
cpt = 0
term_score = {}

# loading patch infos
dct_patch = {}
with open(IMG_DIR + "../tile_selection.tsv") as f:
    rcsv = csv.reader(f, delimiter="\t")

    # read the first line that holds column labels
    csv_labels = rcsv.__next__()
    for record in rcsv:
        if record[3] == '1':
            dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))

mask_type = load_annotations("./annoWeird.csv", "./anno.csv")

ploud = 0
for key, ptch in dct_patch.items():
    # if ploud>100:
    #     break
    # ploud+=1
    # addfigure_xy(ptch.get_pos()[0], 55040-ptch.get_pos()[1], ax, IMG_DIR + key + ".png")
    # TODO: remove the try catch here bc of bad data
    try:
        a = ptch
        sh = shapely.geometry.Polygon(
            [(a.column * a.size, IMG_SIZE[1] - a.row * a.size),
             (a.column * a.size + a.size, IMG_SIZE[1] - a.row * a.size),
             (a.column * a.size + a.size, IMG_SIZE[1] - a.row * a.size + a.size),
             (a.column * a.size, IMG_SIZE[1] - a.row * a.size + a.size)])
        for i, j in mask_type.items():
            if j["WKT"].contains(sh) or j["WKT"].crosses(sh):
                if j["Term"] not in term_score:
                    term_score[j["Term"]] = {}
                    term_score[j["Term"]]["Predicted"] = [[] for i in range(nb_clust)]
                    term_score[j["Term"]]["Real"] = class_color
                    term_score[j["Term"]]["Patches"] = []
                    class_color += 1

                real.append(term_score[j["Term"]]["Real"])
                term_score[j["Term"]]["Patches"].append(key)
                j["Patches"].append(key)

                break
    except KeyError:
        pass
plt.show()
nb_im = 20
fig = plt.figure()
for id, j in mask_type.items():
    print(len(j["Patches"]))
    if len(j["Patches"]) > nb_im:
        a = sample(j["Patches"], k=nb_im)
        for i in range(nb_im):
            ax = fig.add_subplot(1, nb_im, i + 1)
            img = imageio.imread(IMG_DIR + a[i] + ".png")
            plt.imshow(img)
            plt.axis("off")
            ax.set_title(i)
        plt.savefig("./Res_Images/Patches_Region/" + j["Term"] + "_" + id + ".png", format="png")

    plt.show()
    print(cpt)
    print("prepare to crash")
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    hist, xbins, ybins, im = ax.hist2d(pred, real, bins=(nb_clust, len(term_score.keys())), cmap= plt.cm.winter)
    ax.xlabel="pred"
    ax.ylabel="real"
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            ax.text(xbins[j] + 0.5, ybins[i] + 0.5, hist.T[i, j],
                    color="w", ha="center", va="center", fontweight="bold")

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    hist, xbins, ybins, im = ax.hist2d(pred, real, bins=(nb_clust, len(term_score.keys())), cmap=plt.cm.winter)
    ax.xlabel = "pred"
    ax.ylabel = "real"
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            ax.text(xbins[j] + 0.5, ybins[i] + 0.5, hist.T[i, j],
                    color="w", ha="center", va="center", fontweight="bold")



    plt.show()
    cpt = 1
    print(term_score.values())
    for t, sc in term_score.items():
        plt.subplot(1, len(term_score.keys()), cpt)
        plt.hist(sc["Predicted"], bins=nb_clust, density=False)
        plt.title(t)
        cpt+=1
        print(sc)
    plt.show()

    cpt = 1
    limit = 10
    for t, ms in mask_type.items():

        if True: #len(ms["Clust"]) > 20:
            plt.subplot(1, limit, cpt)
            plt.hist(ms["Clust"], bins=nb_clust, density=False)
            plt.title(ms["Term"])
            cpt += 1
        if cpt == limit:
            break
    plt.show()


"""