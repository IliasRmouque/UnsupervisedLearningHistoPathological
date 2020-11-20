
import logging
import string
import sys
import warnings
from argparse import ArgumentParser
import os
import csv
import shapely.wkt
from PIL import Image
from PIL.ImageDraw import ImageDraw
from cytomine import Cytomine
from cytomine.models import AnnotationCollection
from patches import Patch
from shapely import wkt
from shapely.geometry import Polygon
from cytomine.models import (AnnotationCollection, ImageInstance,
                             ImageInstanceCollection)
import requests
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection
from cytomine import Cytomine
import shutil
host = "http://cytomine.icube.unistra.fr"
public_key = "8da00e26-3bcb-4229-b31d-a2b5937c4e5e"  # check your own keys from your account page in the web interface
private_key = "c0018f6a-8aa1-4791-957b-ab72dce4238d"
Dl_Path = "../Repertoir_Travail/annoPoly2"
PROJECT_ID = '1345'
IM_SCALE_FACTOR = 1
DCT= {}
DCT1={}


def get_object_mask_per_image(image_id, project_id, annotation_id,
                              scale_factor, output_filename):
    res = 1
    image_instances = ImageInstanceCollection().fetch_with_filter("project", project_id)
    image_instance = 0
    for im in image_instances:
        if(int(im.id)== int(image_id)):
            image_instance = im

    image_width = int(image_instance.width * scale_factor)
    image_height = int(image_instance.height * scale_factor)

    annotations = AnnotationCollection()
    annotations.project = project_id
    annotations.image = image_id
    annotations.fetch()
    if len(annotations) <= 0:
        return -1
    result_image = Image.new(mode='1', size=(image_width, image_height), color=0)

    somethingannotated = False
    for annotation in annotations:  # for each annotation (i.e. polygone)
        annotation.fetch()

        if not annotation.term:
            #raise ValueError("Annotation %d has not been associated with a term" % annotation.id)
            warnings.warn("Annotation %d has not been associated with a term" % annotation.id)

        # Get the polygon coordinates from cytomine
        if int(annotation_id) in annotation.term:  #  if the object is the one we want to extract
            if annotation.location.startswith("POLYGON") or annotation.location.startswith("MULTIPOLYGON"):
                somethingannotated = True
                if annotation.location.startswith("POLYGON"):
                    label = "POLYGON"
                elif annotation.location.startswith("MULTIPOLYGON"):
                    label = "MULTIPOLYGON"

                coordinatesStringList = annotation.location.replace(label, '')

                if label == "POLYGON":
                    coordinates_string_lists = [coordinatesStringList]
                elif label == "MULTIPOLYGON":
                    coordinates_string_lists = coordinatesStringList.split(')), ((')

                    coordinates_string_lists = [coordinatesStringList.replace('(', '').replace(')', '') for
                                                coordinatesStringList in coordinates_string_lists]

                for coordinatesStringList in coordinates_string_lists:
                    #  create lists of x and y coordinates
                    x_coords = []
                    y_coords = []
                    for point in coordinatesStringList.split(','):
                        point = point.strip(string.whitespace)  # remove leading and ending spaces
                        point = point.strip(string.punctuation) # Have seen some strings have a ')' at the end so remove it
                        x_coords.append(round(float(point.split(' ')[0])))
                        y_coords.append(round(float(point.split(' ')[1])))

                    x_coords_correct_lod = [int(x * scale_factor) for x in x_coords]
                    y_coords_correct_lod = [image_height - int(x * scale_factor) for x in y_coords]
                    coords = [(i, j) for i, j in zip(x_coords_correct_lod, y_coords_correct_lod)]

                    #  draw the polygone in an image and fill it
                    ImageDraw.Draw(result_image).polygon(coords, outline=1, fill=1)

    if somethingannotated:
        result_image.save(output_filename)
        res = 0
    return res


if __name__ == '__main__':
    with  Cytomine(host=host, public_key=public_key, private_key=private_key,
              verbose=logging.INFO) as cytomine:
        annotations = AnnotationCollection()
        annotations.project = 1345
        annotations.image = 10847
        annotations.jobForTermAlgo = 'tumor'
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showTerm = True
        annotations.showGIS = True
        annotations.showImage = True
        annotations.showUser = True
        annotations.fetch( )
        for a in annotations:
            print(a.image, a.term)







            # r = requests.get(url, stream=True)
            # if r.status_code == 200:
            #     with open('mask_6526769.png', 'wb') as f:
            #         r.raw.decode_content = True
            #         print(r.raw)
            #         shutil.copyfileobj(r.raw, f)
            #
            # break
            #
            # # try:
            # #     DCT[a.term[0]] = a.id
            # # except Exception:
            # #     print(a.id)
            # # print(DCT)
            # # with open("./annoWeird.csv", mode='r') as f:
            # #     rcsv = csv.DictReader(f, delimiter="\t")
            # #     line_count = 0
            # #     for record in rcsv:
            # #         if int(record["Id"]) in list(DCT.values()):
            # #             for k, i in DCT.items():
            # #                 if i == int(record["Id"]):
            # #                     DCT[k] = record["Term"]
            # #                     DCT1[record["Term"]] = k
            # # with open("./annoWeird.csv", mode='r') as f:
            # #     rcsv = csv.DictReader(f, delimiter="\t")
            # #     for record in rcsv:
            # #         if record["ImageId"] == '228133':
            # #             DCT[record["Id"]] = DCT1[record["Term"]]
            # # print(DCT, DCT1)
            # # with open("./anno.csv", mode='r') as f:
            # #     rcsv = csv.DictReader(f, delimiter=";")
            # #     for record in rcsv:
            # #
            # #         if record["Image"] == '228133':
            # #
            # #             shape = wkt.loads(record["WKT "])
            # #             anno = Annotation(location=shape.wkt, id_image='6522463')
            # #             if record["ID"] in DCT:
            # #                 print(DCT[record["ID"]])
            # #                 AnnotationTerm(anno.id, DCT[record["ID"]])
            # #             print(anno)


