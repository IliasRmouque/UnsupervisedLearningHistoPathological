import logging

from cytomine import Cytomine
from cytomine.models import ImageInstanceCollection

from utils.path import *
import os
import csv

def get_best_anotated_images(number):
    dict_area = {}
    dict_name= {}
    with open("annoWeird.csv", 'r') as csvfile:
        rcsv = csv.reader(csvfile, delimiter=";")
        csv_labels = rcsv.__next__()
        for record in rcsv:
            if record[5] not in dict_area:
                dict_area[record[5]] = 0
                dict_name[record[5]] = record[6]
            dict_area[record[5]]+=float(record[1])
    dict_area =dict(sorted(dict_area.items(), key=lambda item: item[1]))
    return [(k, dict_name[k]) for k in list(dict_area.keys())[-number:]]






def download_images(best_img):
    """
    Download images
    Usage: download_images()

    Download all the images listed in the file CSV_IMGS in utils.paths
    """
    print("-- Download images")
    with Cytomine(CYTO_HOST, CYTO_PUB_KEY, CYTO_PRV_KEY, verbose=logging.INFO) as cytomine:
            for id, name  in best_img:
                print("\n- Image" +str(name))
                # Paths
                # Directory path

                # Creates the directory
                print("Creating directory")
                os.makedirs(RAW_IMG_DIR, exist_ok=True)

                # Get instance of image with given id
                image_id = int(id)
                image_instances = ImageInstanceCollection().fetch_with_filter("project", PROJECT_ID)
                image_instance = [im for im in image_instances if im.id == image_id]
                image_instance = image_instance[0]

                # Download image
                print("Downloading image")
                image_instance.download(RAW_IMG_DIR + name)



if __name__ == '__main__':
    bes = get_best_anotated_images(5)
    download_images(bes)