from math import ceil

PTCH_DIR="./../Images/A17-4822_-_2019-11-06_14.45.09_tiles/"


# IMG_ID = '228133'
# IMG_NAME = "H110029662_-_2019-02-27_16.20.10"

IMG_ID = '324901'
IMG_NAME = "H110028759_-_2019-02-27_16.12.43"



ptch_size = 128
LOD = 2
IMG_SIZE = [80640, 55040]
IMG_SIZE = [ceil(IMG_SIZE[0]/LOD), ceil(IMG_SIZE[1]/LOD)]
IMG_DIR = './../PyHIST/output/'+ IMG_NAME + '_' + str(ptch_size) + '*' + str(ptch_size) \
          + "_LOD=" + str(LOD) + '/'+ IMG_NAME + '_tiles/'
