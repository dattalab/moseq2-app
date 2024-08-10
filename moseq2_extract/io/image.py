"""
Image reading/writing functionality.
"""

import os
import ast
import json
import numpy as np
import tifffile
from os.path import join, dirname, exists


def read_tiff_files(input_dir):
    """
    Read ROI output results (Tiff files) located in the given input_directory.

    Args:
    input_dir (str): path to directory containing ROI files.

    Returns:
    images (list): list of 2d arrays of the ROIs.
    filenames (list): list of corresponding filenames to each read image.
    """

    images = []
    filenames = []
    for infile in os.listdir(input_dir):
        if infile[-4:] == "tiff":
            im = read_image(join(input_dir, infile))
            if len(im.shape) == 2:
                images.append(im)
            elif len(im.shape) == 3:
                images.append(im[0])
            filenames.append(infile)

    return images, filenames


def write_image(
    filename, image, scale=True, scale_factor=None, frame_dtype="uint16", compress=0
):
    """
    Save image data.

    Args:
    filename (str): path to output file
    image (numpy.ndarray): the (unscaled) 2-D image to save
    scale (bool): flag to scale the image between the bounds of `dtype`
    scale_factor (int): factor by which to scale image
    frame_dtype (str): array data type
    compress (int): image compression level

    """

    file = filename

    metadata = {}

    if scale:
        max_int = np.iinfo(frame_dtype).max

        if not scale_factor:
            # scale image to `dtype`'s full range
            scale_factor = int(
                max_int / (np.nanmax(image) + 1e-25)
            )  # adding very small value to avoid divide by 0
            image = image * scale_factor
        elif isinstance(scale_factor, tuple):
            image = np.float32(image)
            image = (image - scale_factor[0]) / (scale_factor[1] - scale_factor[0])
            image = np.clip(image, 0, 1) * max_int

        metadata = {"scale_factor": str(scale_factor)}

    directory = dirname(file)
    if not exists(directory):
        os.makedirs(directory)

    tifffile.imwrite(
        file, image.astype(frame_dtype), compress=compress, metadata=metadata
    )


def read_image(filename, scale=True, scale_key="scale_factor"):
    """
    Load image data

    Args:
    filename (str): path to output file
    scale (bool): flag that indicates whether to scale image
    scale_key (str): indicates scale factor.

    Returns:
    image (numpy.ndarray): loaded image
    """

    with tifffile.TiffFile(filename) as tif:
        tmp = tif

    image = tmp.asarray()

    if scale:
        image_desc = json.loads(tmp.pages[0].tags["image_description"].as_str()[2:-1])

        try:
            scale_factor = int(image_desc[scale_key])
        except ValueError:
            scale_factor = ast.literal_eval(image_desc[scale_key])

        if type(scale_factor) is int:
            image = image / scale_factor
        elif type(scale_factor) is tuple:
            iinfo = np.iinfo(image.dtype)
            image = image.astype("float32") / iinfo.max
            image = image * (scale_factor[1] - scale_factor[0]) + scale_factor[0]

    return image
