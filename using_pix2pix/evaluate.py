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

generator = keras.models.load_model("saves/save_NEW2B.h5", compile=True)

generator.summary()

text = ["以呂波耳本部止", "千利奴流乎和加", "餘多連曽津祢那", "良牟有為能於久", "耶万計不己衣天", "阿佐伎喩女美之", "恵比毛勢須"]

all_images = Image.new("L", (0, 0), color=255)
all_images_rrpl = Image.new("L", (0, 0), color=255)
all_images_arial = Image.new("L", (0, 0), color=255)

for sentence in text:

    chars = list(sentence)
    images = Image.new("L", (0, 0), color=255)
    images_rrpl = Image.new("L", (0, 0), color=255)
    images_arial = Image.new("L", (0, 0), color=255)

    for char in chars:

        rrpl, arial = data.getCharPair(char)
        output = generator.predict(np.expand_dims(rrpl, 0))

        images = get_concat_h(images, data.drawOutput(output[0]))
        images_rrpl = get_concat_h(images_rrpl, data.drawOutput(rrpl))
        images_arial = get_concat_h(images_arial, data.drawOutput(arial))

    all_images = get_concat_v(all_images, images)
    all_images_rrpl = get_concat_v(all_images_rrpl, images_rrpl)
    all_images_arial = get_concat_v(all_images_arial, images_arial)

all_images.save("test.png")
all_images_rrpl.save("test_rrpl.png")
all_images_arial.save("test_arial.png")
