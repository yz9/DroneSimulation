"""Loader for images.
"""

from os import listdir, makedirs, remove
from os.path import join, exists, isdir
import os.path
import sys
import logging
import numpy as np
import random
from PIL import Image

from sklearn.datasets.base import Bunch
from sklearn.externals.six import b
from sklearn.externals.joblib import Memory

import matplotlib.colors

logger = logging.getLogger(__name__)
debug = 0
MAXFILES=800
usehue = 0    # use summarized hue info
max_image_area = 150*150   # resize images to always be smaller than this (w*h)
standardize_image_size = 0


def scale_image(image):
    """Scale back to 0-1 range in case of normalization for plotting"""
    scaled = image - image.min()
    scaled /= scaled.max()
    return scaled



def _load_imgs(file_paths, slice_, color, resize, hue=0):
    """Internally used to load images"""

    # Try to import imread and imresize from PIL. We do this here to prevent
    # the whole sklearn.datasets module from depending on PIL.
    try:
        try:
            from scipy.misc import imread
        except ImportError:
            from scipy.misc.pilutil import imread
        from scipy.misc import imresize
    except ImportError:
        raise ImportError("The Python Imaging Library (PIL)"
                          " is required to load data from jpeg files")

    # image0 = np.asarray(imread(file_paths[0]), dtype=np.float32)
    # w,h,depth = image0.shape

    # compute the portion of the images to load to respect the slice_ parameter
    # given by the caller
    default_slice = (slice(0, 250), slice(0, 250))
    default_slice = (slice(0, 3000), slice(0, 3000))
    if slice_ is None:
        slice_ = default_slice
        h_slice, w_slice = slice_
        h = (h_slice.stop - h_slice.start) / (h_slice.step or 1)
        w = (w_slice.stop - w_slice.start) / (w_slice.step or 1)
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))
        h = 256
        w = 256


    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    # allocate some contiguous memory to host the decoded image slices
    n_images = len(file_paths)
    # huespace is an array of vectors summarizing hue space
    huespace = np.zeros((n_images, 1, 6), dtype=np.float32)
    if not color:
        # images = np.zeros((n_images, h, w), dtype=np.float32)
        images = []
    else:
        # images = np.zeros((n_images, h, w, 3), dtype=np.float32)
        images = []

    # iterate over the collected file path to load the jpeg files as numpy
    # arrays
    for i, file_path in enumerate(file_paths):
        if (i % 500 == 0) and (i>0):
            logger.info("Loading image #%05d / %05d"% (i + 1, n_images))
        huerange = [0,0,0,0,0,0]
        try: 
            image = np.asarray(imread(file_path)[slice_], dtype=np.float32)
            if debug: print "Loaded image, type:",type(image),image.shape
        except IOError:
            print "**** Cannot open image file",file_path,"***"
            continue
        image /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
        if debug and i<1: print "raw image shape (_load_imgs)",image.shape,"(before slicing), from",file_path
        if resize is not None:
            if standardize_image_size:
                image = imresize(image, (h,w) )  # was resize factor, now use stadard size
            else:
                image = imresize(image, resize )  # was resize factor, now use stadard size
            if i<1: print "resized image shape (_load_imgs)",image.shape,"from",file_path
        else:
            if standardize_image_size:
                image = imresize(image, (h,w) )  # was resize factor, now use stadard size
            w = image.shape[0]
            h = image.shape[1]
            if w*h > max_image_area: # force resize of too big
                print "Resized from",w,h,
                bigger = max(w,h)
                print "(biggest dimension",bigger,")",
                resize = float(256./bigger)
                h = int(resize * h)
                w = int(resize * w)
                print "to",w,h
                image = imresize(image, (h,w) )  # was resize factor, now use stadard size
            
        if not color:
            # average the color channels to compute a gray levels
            # representaion
            if hue:
                # HUE channel:
                img_hsv = matplotlib.colors.rgb_to_hsv(image[...,:3])
                image = img_hsv[...,0]
                if i<1: print "Hue channel (only) kept to get single channel image."
                huerange = image.min(), image.mean(), image.max()
            else:
                if len(image.shape)>=3:
                    if usehue:
                        img_hsv = matplotlib.colors.rgb_to_hsv(image[...,:3])
                        hues = img_hsv[...,0]
                        huerange = np.percentile(hues, [5,20, 80, 95]) + [ hues.mean(), hues.var()  ]
                        # LUMINANCE: 
                    else: huerange = [0,0,0,0,0,0]
                    # Use color average to obtaimn greyscale
                    image[:,:,1] *=0
                    image = image.mean(axis=2)
                    if debug and i<1: print "Averaged color channels to get achromatic luminance image."
                    #image = Image.fromarray(image, 'RGB')
                    #image = image.convert(mode="I", matrix=None, dither=None, palette=0, colors=256*256*256)
                    #image = np.asarray( image )
                    #if debug and i<1: print "Combined color channels to get I-format 32bit image."
                else:
                    if debug or usehue: print file_path, "WARNING: ** not a color image **"
                    huerange = [0,0,0,0,0,0]

        try: images[i, ...] = image  # array
        except: images.append(image) # list
        if usehue: huespace[i, ...] = huerange

    return images, huespace



def _fetch_imagesets(data_folder_path, slice_=None, color=False, resize=None, hue=0,
                      min_images_per_category=0, restrict=None ):
    """Perform the actual data loading for the coral images dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # scan the data folder content to retain people with more that
    # `min_images_per_category` image pictures
    global debug
    category_names, file_paths = [], []
    if restrict == None:
        for category_name in sorted(listdir(data_folder_path)):
            if category_name == ".DS_Store": continue  # skip this utility directory
            if category_name.startswith("skip"): continue
            print "Loading category",category_name  # debug
            folder_path = join(data_folder_path, category_name)
            if not isdir(folder_path):
                continue
            # paths = [join(folder_path, f) for f in listdir(folder_path)]
            cpaths = [ ]
            for fn in listdir(folder_path):
               for imagesuffix in [ ".jpg", ".jpeg", ".png" ]:
                 if fn and fn.lower().endswith(imagesuffix):
                     cpaths.append( "/".join( [ folder_path,fn ]) )
            if debug:
                print "Using only 20 randomly selected files for debug"
                random.shuffle(cpaths)
                cpaths = cpaths[0:20]
            random.shuffle(cpaths)
            cpaths = cpaths[0:MAXFILES]

            n_pictures = len(cpaths)
            print "    _fetch_imagesets: got",n_pictures,"pictures of",category_name
            if n_pictures >= min_images_per_category:
                category_name = category_name.replace('_', ' ')
                category_names.extend([category_name] * n_pictures)
                np.random.RandomState(42).shuffle( cpaths )
                file_paths.extend(cpaths)
    elif restrict == 0:
            category_name = os.path.basename(data_folder_path)
            cpaths = [ ]
            for z,b,c in os.walk( data_folder_path ):
               for imagesuffix in [ ".jpg", ".jpeg", ".png" ]:
                  for fn in c:
                      if fn and fn.lower().endswith(imagesuffix): cpaths.append(  "/".join( [ z,fn ]) )

            if debug:
                print "Using only 30 randomly selected files for debug"
                random.shuffle(cpaths)
                cpaths = cpaths[0:30]
            n_pictures = len(cpaths)
            print "GD: got",n_pictures,"pictures of",category_name
            if n_pictures >= min_images_per_category:
                category_name = category_name.replace('_', ' ')
                category_names.extend([category_name] * n_pictures)
                # np.random.RandomState(42).shuffle( paths )
                file_paths.extend(cpaths)

    n_images = len(file_paths)
    if n_images == 0:
        raise ValueError("min_images_per_category=%d is too restrictive" % min_images_per_category)

    target_names = np.unique(category_names)
    target = np.searchsorted(target_names, category_names)

    images, huespaces = _load_imgs(file_paths, slice_, color, resize, hue=hue)
    if n_images != len(images):  # in case we didn't accept as many as expected
        print "*** WARNING: No all expected image paths were user. ***"
        n_images = len(images)

    # shuffle the images with a deterministic RNG scheme to avoid having
    # all images of the same category in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption
    indices = np.arange(n_images)
    np.random.RandomState(42).shuffle(indices)
    print "Target names",target_names
    # print "Shuffled Indices",indices
    #images, target, huespaces = images[indices], target[indices], huespaces[indices]
    images = [images[i] for i in indices]
    huespaces = huespaces[indices]
    target = target[indices]
    print len(file_paths),"paths and",len(indices),"indices."
    file_paths = [file_paths[i] for i in indices]

    return images, target, target_names, file_paths, huespaces


def fetch_imagesets( data_folder_path = "../coral_labeling/Labels",
                     funneled=True, resize=None,
                     min_images_per_category=0, color=False, hue=0, restrict=None,
                     #slice_=(slice(0, 255), slice(0, 318)),
                     slice_=None,
                     download_if_missing=True):
    """Loader for images.

    This dataset is a collection of JPEG pictures 

    Each pixel of each channel
    (color in RGB) is encoded by a float in range 0.0 - 1.0.

    Parameters
    ----------
    data_home: optional, default: None
        Specify another download and cache folder for the datasets. By default
    funneled: boolean, optional, default: True
        Download and use the funneled variant of the dataset.

    resize: float, optional, default 0.5
        Ratio used to resize the each image picture.

    min_images_per_category: int, optional, default None
        The extracted dataset will only retain pictures of people that have at
        least `min_images_per_category` different pictures.

    color: boolean, optional, default False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than than the shape with color = False.

    slice_: optional
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.
        THIS IS NOT SUPPORTED SICNE IMAGES must ALWAYS COME FROM LOCAL FILES.

    hue: optional, False by default
        Return hsv images rather than intensity or RGB

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (13233, 2914)
        Each row corresponds to a ravelled image image of original size 62 x 47
        pixels.

    dataset.images : numpy array of shape (13233, 62, 47)
        Each row is a image image corresponding to one of the 5749 categories in
        the dataset.

    dataset.target : numpy array of shape (13233,)
        Labels associated to each image image. Those labels range from 0-5748
        and correspond to the category IDs.

    restrict : restrict category selection.  If None, all category directories
        within the data folder are used, if a string, only the specified category 
        is used, if 0 (zero) then files in the main data folder (only) are used, 
        if 1 (unimplemented) all subcategories are merged into one.

    dataset.target_names : names of the categories (folders with images in them)

    dataset.paths : pathnames to the individual images

    dataset.huespaces : array of vectors summarizing hue information

    dataset.DESCR : string
    """
    #images_home = "/Volumes/Macintosh_HD/Users/dudek/Code/coral_labeling/Labelsx"

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    #GD m = Memory(cachedir=images_home, compress=6, verbose=0)
    #GD load_func = m.cache(_fetch_imagesets)

    # load and memoize the pairs as np arrays
    #GD images, target, target_names = load_func(
    images, target, target_names, paths, huespaces = _fetch_imagesets(
        data_folder_path, resize=resize, restrict=restrict,
        min_images_per_category=min_images_per_category, color=color, hue=hue, slice_=slice_)

    # pack the results as a Bunch instance
    # return Bunch(data=images.reshape(len(images), -1), images=images,
    return Bunch(data=images, images=images,
                 target=target, target_names=target_names, paths=paths, huespaces=huespaces,
                 DESCR="coral dataset")


