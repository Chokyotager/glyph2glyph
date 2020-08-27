import tensorflow as tf
from tensorflow import keras

class Model ():

    def __init__ (self):

        input = keras.Input(shape=(128, 128, 1))

        conv_1 = keras.layers.Conv2D(64, 7, strides=3, activation=keras.activations.selu)(input)
        conv_2 = keras.layers.Conv2D(64, 5, strides=3, activation=keras.activations.selu)(conv_1)
        conv_3 = keras.layers.Conv2D(64, 3, strides=3, activation=keras.activations.selu)(conv_2)

        deconv_1 = keras.layers.Conv2DTranspose(64, 3, strides=3, activation=keras.activations.selu)(conv_3)
        deconv_2 = keras.layers.Conv2DTranspose(64, 3, strides=3, activation=keras.activations.selu)(deconv_1)
        output = keras.layers.Conv2DTranspose(64, 3, strides=3, activation=keras.activations.selu)(deconv_2)

        model = keras.Model(inputs=input, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam, loss=tf.losses.mean_pairwise_squared_error, metrics=["mse"])

        self.model = model
