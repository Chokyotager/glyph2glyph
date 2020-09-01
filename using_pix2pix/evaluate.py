import os
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras

from data import Data

data = Data()

def get_concat_h(im1, im2):


    dst = Image.new("L", (im1.width + im2.width, max([im1.height, im2.height])), color=255)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new("L", (max([im1.width, im2.width]), im1.height + im2.height), color=255)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

generator = keras.models.load_model("saves/save.h5", compile=True)

text = ["雪花飄飄北風蕭蕭", "天地一片蒼茫", "一剪寒梅傲立雪中"]

all_images = Image.new("L", (0, 0), color=255)

for sentence in text:

    chars = list(sentence)
    images = Image.new("L", (0, 0), color=255)
    for char in chars:

        rrpl, arial = data.getCharPair(char)
        output = generator.predict(np.expand_dims(rrpl, 0))

        images = get_concat_h(images, data.drawOutput(output[0]))

    all_images = get_concat_v(all_images, images)

all_images.save("test.png")
