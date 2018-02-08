#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:02:24 2018

@author: fredoleary
"""
import os
from enum import Enum
import numpy as np
from scipy import misc

# pylint: disable=C0301
# pylint: disable=C0103

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_DEPTH = 3

THERMAL_HEIGHT = 448
THERMAL_WIDTH = 448
PIXEL_BLOCK = 14

TRAIN_SRC = "train_src"
TEST_SRC = "test_src"


class Label(Enum):
    """
    Thermal image categories
    """
    HEAD = 0
    HAND = 1
    FINGER = 2
    NOISE = 3

def reduce_image(file_in, dir_out, file_out):
    """
    1) load image file, reduce it to 32*32 color image.
    2) Save the image if file_out is not 'None'
    3) Return compressed image where RGB is packed as 0xRRGGBB (32 bit unsigned)
    """

    try:
        image = misc.imread(file_in)
        image_dims = image.shape
        np_ret = None
        if image_dims[0] != IMAGE_HEIGHT or image_dims[1] != IMAGE_WIDTH or image_dims[2] < IMAGE_DEPTH:
            print("Invalid image shape", image.shape)
        else:
            print("Processing", file_in)
            np_out = np.ndarray(shape=(32, 32, 3), dtype=np.int)
            np_ret = np.ndarray(shape=(32, 32), dtype=np.uint32)
            src_row_index = (IMAGE_HEIGHT - THERMAL_HEIGHT)/2
            src_col_index = (IMAGE_WIDTH - THERMAL_WIDTH)/2
            count = int(THERMAL_WIDTH/PIXEL_BLOCK)
            offset = PIXEL_BLOCK/2
            for row in range(count):
                for col in range(count):
                    row_index = int(src_row_index+(row*PIXEL_BLOCK)+offset)
                    col_index = int(src_col_index+(col*PIXEL_BLOCK)+offset)
                    pixel = image[row_index, col_index]
                    np_out[row, col, 0] = pixel[0]
                    np_out[row, col, 1] = pixel[1]
                    np_out[row, col, 2] = pixel[2]
                    ai_value = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2]
                    np_ret[row, col] = ai_value

            # convert to floats
            np_ret = np_ret.astype('float32')
            # convert to range 0.0 -> 1.0
            np_ret /= 0xFFFFFF
            try:
                if file_out is not None:
                    if not os.path.isdir(dir_out):
                        os.makedirs(dir_out)
                    dest_file = dir_out +"//" + file_out
                    misc.imsave(dest_file, np_out)

            except (FileExistsError, PermissionError, NotADirectoryError) as err:
                print("File Error ", err)
                np_ret = None
        return np_ret

    except FileNotFoundError as err2:
        print("File Not Found ", err2.filename)
        return None

def label_image(file_in):
    """
    Return image label, (Head, hand, noise etc)
    """
    if file_in.startswith("head"):
        return Label.HEAD.value
    if file_in.startswith("noise"):
        return Label.NOISE.value
    if file_in.startswith("finger"):
        return Label.FINGER.value
    if file_in.startswith("hand"):
        return Label.HAND.value

    raise ValueError(" Unknown label " + file_in)

def load_data():
    """
    Training data exists in folder train_src as "Label_nnn.bmp". Example "face_001.bmp"
    Test data exists in folder test_src in the same format
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for file in os.listdir(TRAIN_SRC):
        if file.endswith(".bmp"):
            image = reduce_image(TRAIN_SRC + "/" + file, None, None)
            if image is not None:
                x_train.append(image)
                y_train.append(label_image(file))

    for file in os.listdir(TEST_SRC):
        if file.endswith(".bmp"):
            image = reduce_image(TEST_SRC + "/" + file, None, None)
            if image is not None:
                x_test.append(image)
                y_test.append(label_image(file))
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    np_ai = reduce_image("src_images/x001.bmp", "dest_images", "fred.bmp")
    if np_ai is not None:
        print("np_ai - shape", np_ai.shape, "np_ai - dtype", np_ai.dtype)
        np.set_printoptions(formatter={'int':hex})
        print("np_ai data", np_ai)
    print("Done")
