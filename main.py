import csv
import tempfile
from math import ceil

from random import sample
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
from shapely import wkt
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from descartes.patch import PolygonPatch
from matplotlib.colors import hsv_to_rgb

import utils.path as path
from draw_res_image import draw_res_image, draw_origin_image
from patches import Patch
from ensemble import  ensemble_clustering
from data_manipulation import process_data, flatten_raw_data

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



def ensemble_clustering_and_img_drawing(image_size, patch_size, listLod, listClusters, listMethods,  seuil=0.5, terminaison='', namefolder='./',suffixe=''  ):
    for i in range(10):
        dct_lower_res, dimg, color, size = ensemble_clustering(image_size, patch_size, listLod, listClusters, listMethods,  seuil)
        mask_type = load_annotations("./annoWeird.csv", "./anno.csv", path.IMG_ID, color)
        a = data_visualisation(color, mask_type, dct_lower_res, size, min(listLod), path.IMG_ID, terminaison=terminaison,
                       namefolder=namefolder, suffixe=suffixe)

        with open(namefolder + "val.csv", 'a') as f:
            wcsv = csv.writer(f, delimiter="\t")
            # read the first line that holds column labels
            wcsv.writerow(a)
        # clustering_and_img_drawin(size, dct_lower_res, dimg, color, res_name=namefolder + "dessin",
        #                                 thumb_size=(5000, 5000))
    # if ARI<a[0]:
    #     clustering_and_img_drawin(size, dct_lower_res, dimg, color, res_name=namefolder + "dessin",
    #                               thumb_size=(2500, 2500))
    #     ARI=a[0]







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

    # just show the
    fig.set_size_inches(30, 15)
    plt.title("coloration des zones déjà annotées" + terminaison)
    plt.savefig(namefolder + "coloration_zones_annotées_" + terminaison + suffixe + ".png", format="png")
    # ax.cla()

    print("Pour", nb_clust, "clusters \nARI=", adjusted_mutual_info_score(real, pred), "\nNMI=",
          normalized_mutual_info_score(real, pred), "\nhomogenitiy=", )
    predt=[]
    rt=[]
    for i in range(len(pred)):
        if real[i] == term_score["tumor"]["Real"]:
            predt.append(pred[i])
            rt.append(term_score["tumor"]["Real"])
    #True positive False_Positive for tumor


    maxs=[]

    for i in range(nb_clust):
      if i in pred:
        q=(predt.count(i)/pred.count(i))
        if q>0.5:
               maxs.append(i)

    fsc, sens, spec = 0 ,0 ,0
    if maxs:
        cptf=0
        cptt=0
        for i in maxs:
            cptt += predt.count(i)
            cptf += pred.count(i)
        print(maxs)
        print(cptt)
        print(len(predt))
        vp = cptt
        fp = cptf - cptt
        vn = len(pred) - len(predt) - cptf + cptt
        fn = len(predt) - cptt

        spec = vn / (vn + fp)
        sens = vp / (vp + fn)
        fsc =vp/(vp+(fp+fn)/2)

        print("vrai positif:", vp)
        print("faux positif:", fp )
        print("vrai négatif:", vn )
        print("faux négatif:",fn)
        print("sensibilité:", sens)
        print("spécificité:", spec)
        print("fscore:", fsc)





    classesorder ={}
    for k, i in term_score.items():
        classesorder[len(i["Predicted"])] = k
        print(k, len(i["Predicted"]))




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
    return adjusted_mutual_info_score(real, pred), fsc





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
                mask_type[record["Id"]]["WKT"] = Polygon()

    with open(file_with_Polygon, mode='r') as f:
        rcsv = csv.DictReader(f, delimiter=";")
        for record in rcsv:
            try:
                if record["Image"] == img_id:
                    mask_type[record["ID"]]["WKT"] = wkt.loads(record["WKT"])
                    if ax is not None:
                        ax.plot(*wkt.loads(record["WKT"]).exterior.coords.xy)
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
        term_list = get_termlist(dct_patch, mask_type, nb_clust, IMG_SIZE, path.LOD)
        data = np.load(IMG_DIR + data_name, allow_pickle=True)
        raw_data, labels = process_data(data)
        raw_data = flatten_raw_data(raw_data)
        data = dict(zip(labels, raw_data))
        UMAP_for_a_class(data, nb_clust, term_list, axes, len(data_list), shift, nb_limit=100)
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
    namefolder = "./Tableau-Rapport/"
    base= namefolder
    makedirs(namefolder, exist_ok=True)
    dct_patch = {}
    mask_type = load_annotations("./annoWeird.csv", "./anno.csv", path.IMG_ID, 0)
    with open(path.IMG_DIR + "../tile_selection2.tsv") as f:
        rcsv = csv.reader(f, delimiter="\t")
        # read the first line that holds column labels
        csv_labels = rcsv.__next__()
        for record in rcsv:
            if record[3] == '1':
                dct_patch[record[0]] = Patch(record[0], size=path.ptch_size, row=int(record[4]), column=int(record[5]))

    nbc = 10
    minC = 6
    th=0.7
    print("Ensemble2")
    for llod in [[8,8,2]]:
     for nclust in [[4,8,4],[4,8,6],[4,8,8],[6,8,4],[6,8,6],[6,8,8]]:
      for path.IMG_ID, path.IMG_NAME, path.IMG_SIZE in [ ('10823', "A17-4808_-_2019-11-06_14.31.08",[65280 , 53760] ),\
                                        ('6522463', 'H110029662_-_2019-02-27_16.20.10', [84480, 55040] ),\
                                        ('324901','H110028759_-_2019-02-27_16.12.43', [80640, 55040])]:
       for listMethods in [[clustering.make_clusters_KMeans, clustering.make_clusters_KMeans,clustering.make_clusters_KMeans,clustering.make_clusters_KMeans, clustering.make_clusters_KMeans, clustering.make_clusters_KMeans, clustering.make_clusters_KMeans]]:
        with open(namefolder + "val.csv", 'a') as f:
            wcsv = csv.writer(f, delimiter="\t")
            # read the first line that holds column labels
            wcsv.writerow([ str(llod) + " " + str(nclust),path.IMG_NAME, "kmeans clustering"  ])

        ll = llod #* (len(listMethods))
        lc = nclust# * (len(listMethods))
        print(ll,lc)
        ensemble_clustering_and_img_drawing(path.IMG_SIZE, path.ptch_size, listMethods=listMethods, listLod=ll, listClusters=lc,seuil=th, namefolder=namefolder)

    # for i in ['1',"2","3"]:
    #     namefolder = base+"res"+i+"/"
    #     makedirs(namefolder, exist_ok=True)
    #     data = np.load("best_autoenco/res" +i +".npy", allow_pickle=True)
    #     raw_data, labels = process_data(data)
    #     raw_data = flatten_raw_data(raw_data)
    #     labels = labels
    #     data = dict(zip(labels, raw_data))
    #
    #
    #     a,_= clustering.make_clusters_Agglo(raw_data, 16)
    #     print(np.unique(a))
    #     dct_col = dict(zip(labels, a))
    #     for l, c in dct_col.items():
    #         try:
    #          dct_patch[l[:-4]].colour = c
    #         except:
    #             print(l)
    #     clustering_and_img_drawin(IMG_SIZE, dct_patch, IMG_DIR, nb_clust=len(np.unique(a)), res_name=namefolder + 'res' ,
    #                                   thumb_size=(2000, 2000))
    #     data_visualisation(len(np.unique(a)), mask_type, dct_patch, IMG_SIZE, LOD, IMG_ID, '', namefolder)
    #     clustering.plot_dendrogram(raw_data)

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