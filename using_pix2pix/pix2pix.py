# https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py

import tensorflow as tf
from tensorflow import keras

class Pix2Pix ():

    def __init__ (self):

        self.image_rows = 128
        self.image_columns = 128
        self.channels = 1

        # Number of filters in the first layer of G and D
        self.gf = 12
        self.df = 12

        self.image_shape = (self.image_rows, self.image_columns, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.image_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.buildDiscriminator()
        self.discriminator.compile(loss="mse",
            optimizer=optimizer,
            metrics=["accuracy"])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.buildGenerator()

        # Input images and their conditioning images
        img_A = keras.layers.Input(shape=self.image_shape)
        img_B = keras.layers.Input(shape=self.image_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = keras.Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=["mse", "mae"],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def buildGenerator (self):

        def conv2d (layer_input, filters, f_size=3, bn=True):
            d = keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, activation="selu", padding="same")(layer_input)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = keras.layers.UpSampling2D(size=2)(layer_input)
            u = keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding="same", activation="selu")(u)
            if dropout_rate:
                u = keras.layers.Dropout(dropout_rate)(u)
            u = keras.layers.BatchNormalization(momentum=0.8)(u)

            u = keras.layers.Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = keras.layers.Input(shape=self.image_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = keras.layers.UpSampling2D(size=2)(u6)
        output_img = keras.layers.Conv2D(self.channels, kernel_size=4, strides=1, padding="same", activation="tanh")(u7)

        return keras.Model(d0, output_img)

    def buildDiscriminator (self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding="same", activation="selu")(layer_input)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        img_A = keras.layers.Input(shape=self.image_shape)
        img_B = keras.layers.Input(shape=self.image_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = keras.layers.Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding="same")(d4)

        return keras.Model([img_A, img_B], validity)
