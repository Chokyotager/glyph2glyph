# https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

class Pix2Pix ():

    def __init__ (self, hyperparam):

        def wasserstein_loss(y_true, y_pred):
            return keras.backend.mean(y_true * y_pred)

        # Define hyperparams
        initial_lr_gen = hp.Float("lr_gen", min_value=1e-16, max_value=1e-3, step=64)
        initial_lr_discrim = hp.Float("lr_discrim", min_value=1e-16, max_value=1e-3, step=64)
        beta1_gen = hp.Float("beta1_gen", min_value=0, max_value=1, step=10)
        beta1_discrim = hp.Float("beta1_discrim", min_value=0, max_value=1, step=10)
        beta2_gen = hp.Float("beta1_gen", min_value=0, max_value=1, step=10)
        beta2_discrim = hp.Float("beta1_gen", min_value=0, max_value=1, step=10)
        mae_weight = hp.Float("mae_weight", min_value=0.1, max_value=100, step=64)

        self.image_rows = 128
        self.image_columns = 128
        self.channels = 1

        # Number of filters in the first layer of G and D
        self.gf = 24
        self.df = 24

        self.gbn = True
        self.dbn = True

        self.image_shape = (self.image_rows, self.image_columns, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.image_rows / 2**4)
        #self.disc_patch = (patch, patch, 1)

        # Previously: 10x smaller for both
        lr_discrim = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_lr_gen, decay_steps=10000, decay_rate=0.99)
        lr_gen = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_lr_discrim, decay_steps=10000, decay_rate=0.99)

        discrim_optimizer = keras.optimizers.Adam(learning_rate=lr_discrim, beta_1=beta1_gen, beta_2=beta2_gen)
        gen_optimizer = keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta1_discrim, beta_2=beta2_discrim)

        discrim_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # Build and compile the discriminator
        self.discriminator = self.buildDiscriminator()
        self.discriminator.compile(loss=discrim_loss,
            loss_weights=0.5,
            optimizer=discrim_optimizer,
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
        fake_B = self.generator(img_A)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([img_A, fake_B])

        self.combined = keras.Model(inputs=[img_A, img_B], outputs=[valid, img_B])
        self.combined.compile(loss=[gen_loss, "MAE"],
                              loss_weights=[1, mae_weight],
                              optimizer=gen_optimizer)

    def buildGenerator (self):

        def conv2d (layer_input, filters, f_size=7, bn=True):
            d = keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding="same")(layer_input)
            d = keras.layers.PReLU()(d)
            if bn:
                d = keras.layers.BatchNormalization()(d)
                d = keras.layers.LayerNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=7, dropout_rate=0):
            u = keras.layers.UpSampling2D(size=2)(layer_input)
            u = keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(u)
            u = keras.layers.PReLU()(u)
            if dropout_rate:
                u = keras.layers.Dropout(dropout_rate)(u)

            u = keras.layers.BatchNormalization()(u)

            u = keras.layers.Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = keras.layers.Input(shape=self.image_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=self.gbn, f_size=15)
        d2 = conv2d(d1, self.gf*2, bn=self.gbn, f_size=15)
        d3 = conv2d(d2, self.gf*4, bn=self.gbn)
        d4 = conv2d(d3, self.gf*8, bn=self.gbn)
        d5 = conv2d(d4, self.gf*8, bn=self.gbn)
        d6 = conv2d(d5, self.gf*8, bn=self.gbn)
        d7 = conv2d(d6, self.gf*8, bn=self.gbn)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8, dropout_rate=0.05)
        u2 = deconv2d(u1, d5, self.gf*8, dropout_rate=0)
        u3 = deconv2d(u2, d4, self.gf*8, dropout_rate=0)
        u4 = deconv2d(u3, d3, self.gf*4, dropout_rate=0)
        u5 = deconv2d(u4, d2, self.gf*2, dropout_rate=0)
        u6 = deconv2d(u5, d1, self.gf, dropout_rate=0)

        u7 = keras.layers.UpSampling2D(size=2)(u6)
        output_img = keras.layers.Conv2D(self.channels, kernel_size=4, strides=1, padding="same", activation="tanh")(u7)

        return keras.Model(d0, output_img)

    def buildDiscriminator (self):

        def d_layer(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
            d = keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding="same")(layer_input)
            d = keras.layers.PReLU()(d)
            if dropout_rate:
                d = keras.layers.Dropout(dropout_rate)(d)
            if bn:
                d = keras.layers.BatchNormalization()(d)
                d = keras.layers.LayerNormalization()(d)
            return d

        img_A = keras.layers.Input(shape=self.image_shape)
        img_B = keras.layers.Input(shape=self.image_shape)

        #gaussian_B = tf.keras.layers.GaussianNoise(stddev=1.2)(img_B)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = keras.layers.Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=self.dbn, dropout_rate=0.1, f_size=15)
        d2 = d_layer(d1, self.df*2, bn=self.dbn, dropout_rate=0.1)
        d3 = d_layer(d2, self.df*4, bn=self.dbn, dropout_rate=0.1)
        d4 = d_layer(d3, self.df*8, bn=self.dbn, dropout_rate=0.1)

        flatten = keras.layers.Flatten()(d4)

        validity = keras.layers.Dense(1, activation="sigmoid")(flatten)

        return keras.Model([img_A, img_B], validity)
