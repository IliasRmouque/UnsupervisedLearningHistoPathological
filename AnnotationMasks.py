
import logging
import string
import sys
import warnings
from argparse import ArgumentParser
import os
from PIL import Image, ImageDraw
from cytomine.models import  AnnotationCollection,  ImageInstanceCollection
from cytomine import Cytomine

host = "http://cytomine.icube.unistra.fr"
public_key = "8da00e26-3bcb-4229-b31d-a2b5937c4e5e"  # check your own keys from your account page in the web interface
private_key = "c0018f6a-8aa1-4791-957b-ab72dce4238d"


if __name__ == '__main__':
    parser = ArgumentParser(prog="Cytomine Python client example")

    parser.add_argument('--proj_id', dest='id_project',
                        help="The project from which we want the annotations")

    parser.add_argument('--img_id', dest='id_image',
                        help="The images from which we want the annotations")

    parser.add_argument('--dest', dest='dest', required=False,
                        help="Where to store images")

    parser.add_argument('--downscale',dest='down', required=False,
                        help="the downscale factor, 10 if not given")

    params, other = parser.parse_known_args(sys.argv[1:])

    '--proj_id 1345 --img_id 6522463 --dest ./ploud/ --downscale 100'


    with  Cytomine(host=host, public_key=public_key, private_key=private_key,
              verbose=logging.INFO) as cytomine:

        if params.down:
            scale_factor = 1 / int(params.down)
        else :
            scale_factor = 1 / 10
        proj_id = int(params.id_project)
        image_id = int(params.id_image)
        if params.dest:
            os.makedirs(params.dest, exist_ok=True)


        im = ImageInstanceCollection()
        im.project =proj_id
        im.image = image_id
        im.fetch_with_filter("project", proj_id)
        image_width = int(im[0].width)
        image_height = int(im[0].height)
        print(image_height, image_width)

        annotations = AnnotationCollection()
        annotations.project = proj_id
        annotations.image = image_id
        # if needed
        # annotations.user = user_id
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showTerm = True
        annotations.showGIS = True
        annotations.showImage = True
        annotations.showUser = True
        annotations.fetch( )

        dct_anotations = {}
        for a in annotations:
            if len(a.term) == 1:
                term = a.term[0]
                if term not in dct_anotations:
                    dct_anotations[term]=[]
                dct_anotations[term].append(a.location)
            else:
                warnings.warn("Not suited for multiple or no annotation term")
        for t, lanno in dct_anotations.items():
            result_image = Image.new(mode='1', size=( int(image_width* scale_factor), int(image_height * scale_factor)), color=0)
            for pwkt in lanno:
                if pwkt.startswith("POLYGON"):
                    label = "POLYGON"
                elif pwkt.startswith("MULTIPOLYGON"):
                    label = "MULTIPOLYGON"

                coordinatesStringList = pwkt.replace(label, '')

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
                        point = point.strip(
                            string.punctuation)  # Have seen some strings have a ')' at the end so remove it
                        x_coords.append(round(float(point.split(' ')[0])))
                        y_coords.append(round(float(point.split(' ')[1])))

                    x_coords_correct_lod = [int(x * scale_factor) for x in x_coords]
                    y_coords_correct_lod = [image_height * scale_factor - int(x * scale_factor) for x in y_coords]
                    coords = [(i, j) for i, j in zip(x_coords_correct_lod, y_coords_correct_lod)]

                    #  draw the polygone in an image and fill it
                    ImageDraw.Draw(result_image).polygon(coords, outline=1, fill=1)

            result_image.save(params.dest + '/' + str(t)+'.png')







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


