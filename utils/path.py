from math import ceil

PTCH_DIR="./../Images/A17-4822_-_2019-11-06_14.45.09_tiles/"


# IMG_ID = '228133'
# IMG_NAME = "H110029662_-_2019-02-27_16.20.10"

global IMG_ID, IMG_NAME
IMG_ID, IMG_NAME = ('10823', "A17-4808_-_2019-11-06_14.31.08")
List_IMAGES =   ('10823', "A17-4808_-_2019-11-06_14.31.08"),\
                ('10931', 'A17-7275_-_2019-11-06_15.25.13'),\
                ('10959', 'A18-3898_-_2019-11-06_17.32.26'),\
                ('11213','A19-4118_-_2019-11-06_20.39.42'),\
                ('11207', 'A19-5206_-_2019-11-06_21.00.21'),\
                ('11201', 'H0911305-I1_-_2019-03-08_09.01.07' ),\
                ('324901','H110028759_-_2019-02-27_16.12.43'), \
                ('6522463', 'H110029662_-_2019-02-27_16.20.10'),\
                ('398114','H120004420_-_2019-02-27_21.55.13'),

"""
('10823', "A17-4808_-_2019-11-06_14.31.08"),  #dense que j'ai annoté
    ('10931', 'A17-7275_-_2019-11-06_15.25.13'),
    ('10959', 'A18-3898_-_2019-11-06_17.32.26'),
    ('11213','A19-4118_-_2019-11-06_20.39.42'),
    ('11207', 'A19-5206_-_2019-11-06_21.00.21'),
    ('11201', 'H0911305-I1_-_2019-03-08_09.01.07' ),
    ('324901','H110028759_-_2019-02-27_16.12.43') #le dindon
    ('6522463', 'H110029662_-_2019-02-27_16.20.10'), #la dense avec le cercle bleu
    ('398114','H120004420_-_2019-02-27_21.55.13'), #deuxième plus annotée
    ('11195', 'H1225579-I1_-_2019-03-07_17.37.43') #lots of tissue
"""

ptch_size = 128
LOD = 2
global IMG_SIZE
IMG_SIZE = [80640, 55040]

IMG_DIR = './../Images/Patches/'+ IMG_NAME + '_' + str(ptch_size)\
          + "_LOD=" + str(LOD) + '/'+ IMG_NAME + '_tiles/'

def get_ALL_PATCHES_DIR(lod, size):
    return ALL_PAtches +"_"+str(size)+"_"+str(lod)+'/'

RAW_IMG_DIR = "../Images/Raw_Img/"
DIR_PATCHES = "../Images/Patches/" # + SIZE
ALL_PAtches="../Images/Patches/All"



CYTO_HOST = "http://cytomine.icube.unistra.fr"
CYTO_PUB_KEY = "8da00e26-3bcb-4229-b31d-a2b5937c4e5e"  # check your own keys from your account page in the web interface
CYTO_PRV_KEY = "c0018f6a-8aa1-4791-957b-ab72dce4238d"
PROJECT_ID= 1345