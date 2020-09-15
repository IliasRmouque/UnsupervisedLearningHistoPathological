# -*- coding: utf-8 -*-
#  Odyssee Merveille 08/11/17

import numpy
import openslide
from utils import forcedImageSvs
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import skimage.transform
import warnings
import keras.backend as K
from utils.colourconv import rgb2hed
from skimage import img_as_ubyte
import os
import h5py
from utils.multiclass_downsampling import countless
import math
from skimage.color import rgb2grey
#from stain_normalisation.normalization.reinhard import stainNorm_Reinhard
import pyvips


def read_image(imagePath, dtype=None):
    """
    read_image: read a 2D image and return a numpy array
	Several image formats are allowed (see PIL library)

    :param imagePath: (string) the path of the image to load
    :param dtype: (string) override the image data type
    :return: (numpy.array) the image converted to a numpy array
    """

    format_to_dtype = {
        'uchar': numpy.uint8,
        'char': numpy.int8,
        'ushort': numpy.uint16,
        'short': numpy.int16,
        'uint': numpy.uint32,
        'int': numpy.int32,
        'float': numpy.float32,
        'double': numpy.float64,
        'complex': numpy.complex64,
        'dpcomplex': numpy.complex128,
    }

    img = pyvips.Image.new_from_file(imagePath, access='sequential')

    if not dtype:
        dtype = format_to_dtype[img.format]

    if img.bands > 1:
        return numpy.ndarray(buffer=img.write_to_memory(), dtype=dtype, shape=[img.height, img.width, img.bands])
    else:
        return numpy.ndarray(buffer=img.write_to_memory(), dtype=dtype, shape=[img.height, img.width])


def read_binary_image(imagePath):
    """

    read_binary_image: read a 2D image and convert its values to Boolean

    :param imagePath: (string) the path of the image to load
    :return:  (numpy.array) the image converted to a boolean numpy array
    """

    return read_image(imagePath, dtype=numpy.bool_)


def save_image(array, savePath):
    """
    save_image: save a 2D numpy array as an image
    Several image format are allowed (see PIL library)

    :param array: (numpy.array) the image to save
    :param savePath: (string) the output path
    :return: 0
    """

    dtype_to_format = {
        'uint8': 'uchar',
        'int8': 'char',
        'uint16': 'ushort',
        'int16': 'short',
        'uint32': 'uint',
        'int32': 'int',
        'float32': 'float',
        'float64': 'double',
        'complex64': 'complex',
        'complex128': 'dpcomplex',
    }

    if len(array.shape) == 3:
        img = pyvips.Image.new_from_memory(array.tobytes(order='C'), array.shape[1], array.shape[0], array.shape[2], dtype_to_format[str(array.dtype)])
    elif len(array.shape) == 2:
        img = pyvips.Image.new_from_memory(array.tobytes(order='C'), array.shape[1], array.shape[0], 1, dtype_to_format[str(array.dtype)])
    else:
        raise ValueError('Image array should have 2 (y, x) or 3 (y, x, b) dimensions')

    img.write_to_file(savePath)

    return 0


def save_binary_image(array, savePath):
    """
    save_image: save a 2D binary numpy array
    Several image format are allowed (see PIL library)

    :param array: (numpy.array) the image to save
    :param savePath: (string) the output path
    :return: 0
    """

    return save_image(array.astype(numpy.uint8), savePath)


def convert_image(sourcePath, targetPath, removeSource=False):
    """
    convert_image: convert an image file to another format
    Several image format are allowed (see PIL library)

    :param sourcePath: (string) the image to convert
    :param targetPath: (string) the output path
    """

    img = pyvips.Image.new_from_file(sourcePath, access='sequential')
    img.write_to_file(targetPath)

    if removeSource:
        os.remove(sourcePath)


def read_svs_image(imagePath, lod):
    """
     read_svs_image: read a 2D RGB svs image at a given lod and return a numpy array

    :param imagePath: (string) the path of the svs image
    :param lod: (int) the level of detail to be loaded
    :return: (numpy.array) the image converted to a numpy array (size dependent upon the lod)
    """

    Image.MAX_IMAGE_PIXELS = None

    scaleFactor = 2 ** lod
    imageSvs = openslide.OpenSlide(imagePath)

    vecOfScales = numpy.asarray(imageSvs.level_downsamples)

    if numpy.where(vecOfScales.astype(int) == scaleFactor)[0].size > 0:  #  If the scale exist
        #  Find the index of the given scaleFactor in the vector of scales
        level = numpy.where(vecOfScales.astype(int) == scaleFactor)[0][0]
    else:
        string = [str(int(a)) for a in imageSvs.level_downsamples]
        raise ValueError(
            'The svs image does not contain an image with scaleFactor %i \n scales are: %s' % (scaleFactor, string))

    image = imageSvs.read_region((0, 0), level, imageSvs.level_dimensions[level])
    image = image.convert('RGB')
    imageSvs.close()

    return numpy.asarray(image)


def read_svs_image_forced(imagePath, lod):
    """
    read_svs_image_forced: read a 2D RGB svs at a given lod. If the svs does not contain the requested lod, read a lower
    lod and resize the image to return the image with the correct lod (in this case a warning is issued).

    :param imagePath: (string) the path of the svs image
    :param lod: (int) the level of detail to be loaded
    :return: (numpy.array) the image converted to a numpy array (size dependent upon the lod)
    """

    Image.MAX_IMAGE_PIXELS = None

    scaleFactor = 2 ** lod

    imageSvs = openslide.OpenSlide(imagePath)

    vecOfScales = numpy.asarray(imageSvs.level_downsamples)

    if numpy.where(vecOfScales.astype(int) == scaleFactor)[0].size > 0:  #  If the scale exist
        level = numpy.where(vecOfScales.astype(int) == scaleFactor)[0][0]  #  Find the indice of the given scaleFactor in the vector of scales.
        image = imageSvs.read_region((0, 0), level, imageSvs.level_dimensions[level])
        image = numpy.asarray(image.convert('RGB'))
    else:
        string = [str(int(a)) for a in imageSvs.level_downsamples]
        warnings.warn(
            'The svs image does not contain an image with scaleFactor %i \n\t scales are: %s' % (scaleFactor, string))

        # Find the higher lod which is smaller than "lod"
        lowerScaleFactor = -1
        for indice in range(vecOfScales.size):
            currentScaleFactor = vecOfScales[indice]
            if (currentScaleFactor > lowerScaleFactor and currentScaleFactor < scaleFactor):
                lowerScaleFactor = currentScaleFactor
                indiceLowerScaleFactor = indice

        if lowerScaleFactor == -1:
            raise ValueError(
                'The svs image does not contain an image with scaleFactor %i and no lower scale factor to interpolate it: %s' % (
                scaleFactor, string))

        print("\t Rescaling from scaleFactor %i" % lowerScaleFactor)
        # Read the image at the lower lod
        imageLowerLod = imageSvs.read_region((0, 0), indiceLowerScaleFactor,
                                             imageSvs.level_dimensions[indiceLowerScaleFactor])
        imageLowerLod = numpy.asarray(imageLowerLod.convert('RGB'))
        dimY, dimX = imageSvs.level_dimensions[indiceLowerScaleFactor]

        #  Rescale it
        image = skimage.transform.resize(imageLowerLod, (
            int(dimX / (scaleFactor / lowerScaleFactor)), int(dimY / (scaleFactor / lowerScaleFactor)), imageLowerLod.shape[2]),
                                         mode='reflect', preserve_range=True)
        image = image.astype(numpy.uint8)

    imageSvs.close()

    return image


def open_svs_image_forced(imagePath):
    """
    read_svs_image_forced: read a 2D RGB svs at a given lod. If the svs does not contain the requested lod, read a lower
    lod and resize the image to return the image with the correct lod (in this case a warning is issued).

    :param imagePath: (string) the path of the svs image
    :param lod: (int) the level of detail to be loaded
    :return: (numpy.array) the image converted to a numpy array (size dependent upon the lod)
    """

    Image.MAX_IMAGE_PIXELS = None

    return forcedImageSvs.ForcedOpenSlide(imagePath)


def read_svs_patch_forced(imagePath, lod, location, size):
    """
    read_svs_image_forced: read a 2D RGB svs at a given lod. If the svs does not contain the requested lod, read a lower
    lod and resize the image to return the image with the correct lod (in this case a warning is issued).

    :param imagePath: (string) the path of the svs image
    :param lod: (int) the level of detail to be loaded
    :return: (numpy.array) the image converted to a numpy array (size dependent upon the lod)
    """

    Image.MAX_IMAGE_PIXELS = None

    svsImage = open_svs_image_forced(imagePath)

    patch = svsImage.read_region(location, lod, size)

    svsImage.close()

    return patch


def is_lod_in_svs(imagePath, lod):
    """
    is_lod_in_svs: true if the image contains the given level of detail

    :param imagePath: (string) the path of the svs images
    :param lod: (int) the level of detail required
    :return: (boolean) return true if the svs contains the lod, false otherwise.

    """

    scaleFactor = 2 ** lod

    imageSvs = openslide.OpenSlide(imagePath)

    vecOfScales = numpy.asarray(imageSvs.level_downsamples)

    if numpy.where(vecOfScales.astype(int) == scaleFactor)[0].size > 0:  #  If the scale exist
        return True
    else:
        return False


def save_plot_image(im, savePath, title="", colormap="gray"):
    """

    save_plot_image: save a matplotlib plot of a numpy array

    :param im: (convertible to numpy.array) the data to transform
    :param savePath: (string) the output path
    :param title: (string) the title of the plot (default="")
    :param colormap: (string) the colormap used to generate the plot (default="gray")
    """

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    if not is_numpy_array(im):
        im = numpy.asarray(im)

    plt.imshow(im, colormap)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(savePath)


def show_image(im, title="", colormap="gray", save=""):
    """

    show_image

    :param im: (convertible to numpy.array) the data to transform
    :param title: (string) the name of the plot (default="")
    :param colormap: (string) the colormap used to generate the plot (default="gray")
    :param save: (string) the path to save the output, if the string is the file is not written
    """

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    if not is_numpy_array(im):
        im = numpy.asarray(im)

    plt.imshow(im, colormap)
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)

    if save != "":
        plt.savefig(save)


def is_numpy_array(obj):
    """
    is_numpy_array: check if the object is a numpy array

    :param obj: () the object to check
    :return: (Boolean) True if the obj is a numpy array
    """

    return isinstance(obj, (numpy.ndarray, numpy.generic))


def normalise_image(image):
    """
    normalise_image: normalise image based on its min and max histogram values

    :param image: (numpy.array) the image to normalise
    :return: (numpy.array) the normalised image
    """

    minValue = numpy.amin(image)
    maxValue = numpy.amax(image)

    normalisedImage = (image - minValue) / float(maxValue - minValue)

    return normalisedImage


def unflattenmask(img, patch_size, class_number):
    if K.backend() == 'tensorflow':
        return img.reshape(patch_size, patch_size, class_number, order='A')
    elif K.backend() == 'theano':
        return img.reshape(patch_size, patch_size, class_number, order='C')


def unflattenmasks(imgs, patch_size, class_number):
    number_of_images = imgs.shape[0]

    imgs_reshaped = numpy.empty((number_of_images, patch_size, patch_size, class_number), dtype=imgs.dtype)
    for i in range(number_of_images):
        imgs_reshaped[i] = unflattenmask(imgs[i], patch_size, class_number)

    return imgs_reshaped


def flattenmask(img):
    if K.backend() == 'tensorflow':
        return img.reshape(img.shape[0] * img.shape[0], img.shape[2], order='A')
    elif  K.backend() == 'theano':
        return img.reshape(img.shape[0] * img.shape[0], img.shape[2], order='C')


def flattenmasks(imgs):
    number_of_images = imgs.shape[0]
    input_patch_size = imgs.shape[1]
    class_number = imgs.shape[3]

    imgs_reshaped = numpy.empty((number_of_images, input_patch_size * input_patch_size, class_number), dtype=imgs.dtype)

    for i in range(number_of_images):
        imgs_reshaped[i] = flattenmask(imgs[i])

    return imgs_reshaped


def getreshapeorder():
    if K.backend() == 'tensorflow':
        return 'A'
    elif K.backend() == 'theano':
        return 'C'


def normalise_bw_image_from_norm_stats(sourceimage, t_counts, t_values):
    """
    normalise_bw_image_from_norm_stats: adjust the pixel values of a grayscale image such that its histogram matches
    that of a target image

    :param sourceimage: (numpy.ndarray) image to transform; the histogram is computed over the flattened array
    :param t_counts: TODO
    :param t_values:
    :return: (numpy.ndarray) the transformed output image
    """

    oldshape = sourceimage.shape
    olddtype = sourceimage.dtype

    sourceimage = sourceimage.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    _, bin_idx, s_counts = numpy.unique(sourceimage, return_inverse=True, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_counts = numpy.cumsum(s_counts).astype(numpy.float64)
    s_counts /= s_counts[-1]
    s_counts *= 255

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = numpy.interp(s_counts, t_counts, t_values)

    return interp_t_values[bin_idx].reshape(oldshape).astype(olddtype)


def normalise_rgb_image_from_stat_file(sourceimage, h5py_norm_filename, normalise_within_roi, roimaskfilename=None):
    """
    normalise_rgb_image_from_image: adjust the pixel values of an RGB image such that the histogram of each channel
    matches that of a target image

    :param sourceimage: (numpy.ndarray) image to transform; the histogram is computed over the flattened array
    :param templatefilename: (string) Image to transform; the histogram is computed over the flattened array of
    each channel
    :param normalise_within_roi: (boolean) normalise within a specific region of the image or not
    :param lod: the level of detail of the images
    :param roimaskpath: the mask specifying the region in which to normalise the image
    :return: (numpy.array) the transformed output image
    """

    n = stainNorm_Reinhard.Normalizer()
    n.read_fit(h5py_norm_filename)

    if normalise_within_roi:
        roi_mask = read_binary_image(roimaskfilename)
        sourceimage = n.transform(sourceimage[roi_mask])
    else:
        sourceimage = n.transform(sourceimage)

    return sourceimage


def write_normalisation_data(templatefilename, normalise_within_roi, lod, outputfilename, roimaskfilename=None):

    targetimage = read_svs_image_forced(templatefilename, lod)

    n = stainNorm_Reinhard.Normalizer()

    if normalise_within_roi:
        roi_mask = read_binary_image(roimaskfilename)
        n.fit(targetimage[roi_mask])
    else:
        n.fit(targetimage)

    if os.path.isfile(outputfilename):
        os.remove(outputfilename)

    n.write_fit(outputfilename)


def getpatchsize(base_patch_size, lod):
    """

    getpatchsize get the patch size depending on its base size and the level of detail used

    :param base_patch_size: (int) the size of the patch on the maximum lod size image
    :param lod: (int) the level of detail used for the image
    :return: (int) the real size of the pixels
    """

    return base_patch_size // (2 ** lod)


def downsample_gt(image, target_size):

    image_size = image.shape[0]

    repetitions = int(math.log2(image_size) - math.log2(target_size))

    for _ in range(repetitions):
        image = countless(image)

    return image


def read_segmentations(segmentationpath, image, classes, ordering=None):
    """
    read_segmentations: read all the class probability maps for an image

    :param segmentationpath: (string) the segmentation path
    :param image: (string) the name of the image
    :param classes: (list of string) the list of classes to load, each should be the name of a sub-directory in
    segmentationpath
    :param ordering: (list of int) the order in which to reshuffle the segmentation
    :return: (numpy.array) a 3D array in which the 3rd dimension represents each class
    """

    for i, c in enumerate(classes):
        if i == 0:
            segmentations = read_image(os.path.join(segmentationpath, c, image))
        else:
            segmentations = numpy.dstack((segmentations, read_image(os.path.join(segmentationpath, c, image))))

    if ordering:
        segmentations = segmentations[:, :, ordering]

    return segmentations


def read_segmentations_memmap(segmentationpath, image, classes, ordering=None):
    """
    read_segmentations: read all the class probability maps for an image

    :param segmentationpath: (string) the segmentation path
    :param image: (string) the name of the image
    :param classes: (list of string) the list of classes to load, each should be the name of a sub-directory in
    segmentationpath
    :param ordering: (list of int) the order in which to reshuffle the segmentation
    :return: (numpy.array) a 3D array in which the 3rd dimension represents each class
    """

    import skimage.external.tifffile

    tmpfilenames = []

    for i, c in enumerate(classes):

        filename, file_extension = os.path.splitext(os.path.join(segmentationpath, c, image))

        if not file_extension == '.tif' or file_extension == '.tiff':
            if not os.path.isfile(os.path.join(tempfile.gettempdir(), os.path.basename(filename) + '_' + c + '.tif')):
                convert_image(os.path.join(os.path.join(segmentationpath, c, image)),
                              os.path.join(tempfile.gettempdir(), os.path.basename(filename) + '_' + c + '.tif'),
                              removeSource=False)
                tmpfilenames.append(os.path.join(tempfile.gettempdir(), os.path.basename(filename) + '_' + c + '.tif'))
            filename = os.path.join(tempfile.gettempdir(), os.path.basename(filename) + '_' + c)
            file_extension = '.tif'
        else:
            filename, file_extension = os.path.splitext(os.path.join(segmentationpath, c, image))

        with skimage.external.tifffile.TiffFile(filename + file_extension) as tif:
            if i == 0:
                segmentations = tif.asarray(memmap=True)
            else:
                segmentations = numpy.dstack((segmentations, tif.asarray(memmap=True)))

    if ordering:
        segmentations = segmentations[:, :, ordering]

    return segmentations,


def open_segmentations(segmentationpath, image, classes, ordering=None):
    """
    read_segmentations: read all the class probability maps for an image

    :param segmentationpath: (string) the segmentation path
    :param image: (string) the name of the image
    :param classes: (list of string) the list of classes to load, each should be the name of a sub-directory in
    segmentationpath
    :param ordering: (list of int) the order in which to reshuffle the segmentation
    :return: (numpy.array) a 3D array in which the 3rd dimension represents each class
    """

    for i, c in enumerate(classes):
        if i == 0:
            segmentations = read_image(os.path.join(segmentationpath, c, image))
        else:
            segmentations = numpy.dstack((segmentations, read_image(os.path.join(segmentationpath, c, image))))

    if ordering:
        segmentations = segmentations[:, :, ordering]

    return segmentations


def image_colour_convert(img, color_mode='rgb'):
    if color_mode == 'rgb':
        return img
    elif color_mode == 'greyscale':
        return rgb2grey(img / 255)[..., None]
    elif color_mode == 'haemotoxylin':
        return rgb2hed(img / 255)[..., [0]]
