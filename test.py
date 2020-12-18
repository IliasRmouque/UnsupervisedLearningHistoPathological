
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from argparse import ArgumentParser

import os
import csv

import shapely.wkt





from cytomine import Cytomine
from cytomine.models import AnnotationCollection
from patches import Patch
from utils.path import *

host = "http://cytomine.icube.unistra.fr"
public_key = "8da00e26-3bcb-4229-b31d-a2b5937c4e5e"  # check your own keys from your account page in the web interface
private_key = "c0018f6a-8aa1-4791-957b-ab72dce4238d"

term ={
    ''
}

if __name__ == '__main__':
    with Cytomine(host=CYTO_HOST, public_key=CYTO_PUB_KEY, private_key=CYTO_PRV_KEY,
                    verbose=logging.INFO) as cytomine:
          annotations = AnnotationCollection()
          annotations.project = "1345"
          annotations.showWKT = True
          annotations.showMeta = True
          annotations.showTerm = True
          annotations.showGIS = True
          annotations.fetch()
          print(annotations)

          f = open("./anno.csv", "w+")
          f.write("ID;Image;Project;Term;User;Area;Perimeter;WKT;TRACK \n")
          for annotation in annotations:
                   f.write("{};{};{};{};{};{};{};{}\n".format(annotation.id, annotation.image, annotation.project,
                                                            annotation.term, annotation.user, annotation.area,
                                                           annotation.perimeter, annotation.location))

    exit(0)
    import time
    from main import *

    img_id = '324901'
    IMG_SIZE = [80640, 55040]

    IMG_NAME = "H110028759_-_2019-02-27_16.12.43"
    mask_type = load_annotations("./annoWeird.csv", "./anno.csv", img_id, 0)
    for ptch_size in [64, 128,256]:
        for LOD in [1,2,4,8,16]:
            IMG_SIZE = [80640, 55040]
            dct_patch = {}

            print(LOD, ' ', ptch_size)
            try:
                IMG_DIR = './../PyHIST/output/' + IMG_NAME + '_' + str(ptch_size) + '*' + str(ptch_size) \
                          + "_LOD=" + str(LOD) + '/' + IMG_NAME + '_tiles/'


                IMG_SIZE = [ceil(IMG_SIZE[0]/LOD), ceil(IMG_SIZE[1]/LOD)]



                with open(IMG_DIR + "../tile_selection2.tsv") as f:
                    rcsv = csv.reader(f, delimiter="\t")
                    # read the first line that holds column labels
                    csv_labels = rcsv.__next__()
                    for record in rcsv:
                        if record[3] == '1':
                            dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))
                fig, ax = plt.subplots()
                # mask_type = load_annotations("./annoWeird.csv", "./anno.csv", img_id, 0, ax)
                print(len(dct_patch.items()))

                term_score = {}
                patch = {}

                for name, ptch in dct_patch.items():

                    sh = shapely.geometry.Polygon(
                        [(ptch.column * ptch.size * LOD, IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD),
                         (ptch.column * ptch.size * LOD + ptch.size * LOD, IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD),
                         (ptch.column * ptch.size * LOD + ptch.size * LOD,
                          IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD + ptch.size * LOD),
                         (ptch.column * ptch.size * LOD, IMG_SIZE[1] * LOD - ptch.row * ptch.size * LOD + ptch.size * LOD)])
                    # ax.add_patch(PolygonPatch(sh, color=hsv_to_rgb([1 / 2, 1, 1])))
                    for i, j in mask_type.items():
                        if j["WKT"].contains(sh):
                            if j['Term'] not in patch:
                                patch[j['Term']]=[name]
                            patch[j['Term']].append(name)
                # plt.show()
                a=[]
                print((list(patch)))
                for k, v in patch.items():
                    for p in v: a.append((k, p))

                if len(a) > 100:
                    print(len(a))
                    os.mkdir('./patch_test/patch_size='+str(ptch_size)+'_LOD='+str(LOD)+'/')
                    for idx in range(4):

                            r = sample(a, k=100)
                            fig, ax = plt.subplots()
                            fig.set_size_inches(ptch_size/4, ptch_size/4)
                            for i in range(100):
                                ax = plt.subplot(10, 10, i+1)
                                img = imageio.imread(IMG_DIR + r[i][1] +".png")
                                plt.imshow(img)
                                plt.title("")
                                plt.axis("off")
                            plt.savefig('./patch_test/patch_size='+str(ptch_size)+'_LOD='+str(LOD)+'/'+ str(idx) +"_test_patch.png", format="png")
                            for i in range(100):
                                ax = plt.subplot(10, 10, i+1)
                                img = imageio.imread(IMG_DIR + r[i][1]+".png")
                                plt.imshow(img)
                                plt.title(r[i][0])
                                plt.axis("off")
                            plt.savefig('./patch_test/patch_size='+str(ptch_size)+'_LOD='+str(LOD)+'/'+str(idx) +"_res_patch.png", format="png")
                else:
                    a = []
                    for k, v in dct_patch.items():
                        a.append(k)
                    print(len(a))
                    os.mkdir('./patch_test/patch_size=' + str(ptch_size) + '_LOD=' + str(LOD) + '/')
                    for idx in range(4):
                        r = sample(a, k=100)
                        fig, ax = plt.subplots()
                        fig.set_size_inches(ptch_size/4, ptch_size/4)
                        for i in range(100):
                            ax = plt.subplot(10, 10, i + 1)
                            img = imageio.imread(IMG_DIR + r[i] + ".png")
                            plt.imshow(img)
                            plt.title("")
                            plt.axis("off")
                        plt.savefig('./patch_test/patch_size=' + str(ptch_size) + '_LOD=' + str(LOD) + '/' + str(
                            idx) + "_no_anno_patch.png", format="png")






            except FileNotFoundError as e:
                print(e)
                pass












    # dct_patch = {}
    # ptch_size = 64
    #
    # Dl_Path = "../Repertoir_Travail/annoPoly2"
    #
    # term_score={}
    # mask_type={}
    #
    #
    # with Cytomine(host=host, public_key=public_key, private_key=private_key,
    #               verbose=logging.INFO) as cytomine:
    #     annotations = AnnotationCollection()
    #     annotations.project = "1345"
    #     annotations.showWKT = True
    #     annotations.showMeta = True
    #     annotations.showTerm = True
    #     annotations.showGIS = True
    #     annotations.fetch()
    #     print(annotations)
    #
    #     if Dl_Path:
    #         f = open(Dl_Path + ".csv", "w+")
    #         f.write("ID;Image;Project;Term;User;Area;Perimeter;WKT;TRACK \n")
    #         for annotation in annotations:
    #             if str(annotation.image) == '228133':
    #                 f.write("{};{};{};{};{};{};{};{}\n".format(annotation.id, annotation.image, annotation.project,
    #                                                        annotation.term, annotation.user, annotation.area,
    #                                                        annotation.perimeter, annotation.location))
    #
    #         if str(annotation.image) == '228133':
    #             print(
    #                 "ID: {} | Image: {} | Project: {} | Term: {} | User: {} | Area: {} | Perimeter: {} | WKT: {}".format(
    #                     annotation.id,
    #                     annotation.image,
    #                     annotation.project,
    #                     annotation.term,
    #                     annotation.user,
    #                     annotation.area,
    #                     annotation.perimeter,
    #                     annotation.location
    #                 ))
    #
    #             try:
    #                 # max_size is set to 512 (in pixels). Without max_size parameter, it download a dump of the same size that the annotation.
    #                 annotation.dump(dest_pattern=os.path.join(Dl_Path, "{project}", "crop", "{id}.jpg"),
    #                                 max_size=512)
    #
    #                 annotation.dump(dest_pattern=os.path.join(Dl_Path, "{project}", "mask", "{id}.jpg"),
    #                                 mask=True, max_size=512)
    #                 annotation.dump(dest_pattern=os.path.join(Dl_Path, "{project}", "alpha", "{id}.png"),
    #                                 mask=True, alpha=True, max_size=512)
    #             except Exception as e:
    #                 print("erreur",e)
    #                 pass





    # with open(IMG_DIR + "../tile_selection.tsv") as f:
    #     rcsv = csv.reader(f, delimiter="\t")
    #
    #     # read the first line that holds column labels
    #     csv_labels = rcsv.__next__()
    #
    #     for record in rcsv:
    #         if record[3] == '1':
    #             dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]), column=int(record[5]))
    #
    # with open("./anno.csv") as f:
    #     rcsv = csv.DictReader(f,delimiter=";" )
    #     print(rcsv.fieldnames)
    #     for record in rcsv:
    #         try:
    #             if record["Image"] == '10847':
    #                 mask_type[record["ID"]]["WKT"] = shapely.wkt.loads(record["WKT "])
    #         except:
    #             pass
    # print(len(mask_type))

