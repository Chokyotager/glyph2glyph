import tensorflow as tf
from tensorflow import keras

class Model ():

    def __init__ (self):

        input = keras.Input(shape=(128, 128, 1))

        conv_1 = keras.layers.Conv2D(64, 7, strides=3, activation=keras.activations.selu, padding="valid")(input)
        conv_2 = keras.layers.Conv2D(64, 5, strides=3, activation=keras.activations.selu, padding="valid")(conv_1)
        conv_3 = keras.layers.Conv2D(64, 3, strides=3, activation=keras.activations.selu, padding="valid")(conv_2)

        deconv_1 = keras.layers.Conv2DTranspose(64, 3, strides=3, activation=keras.activations.selu, padding="valid")(conv_3)
        deconv_2 = keras.layers.Conv2DTranspose(64, 5, strides=3, activation=keras.activations.selu, padding="valid")(deconv_1)
        deconv_3 = keras.layers.Conv2DTranspose(64, 7, strides=3, activation=keras.activations.selu, padding="valid")(deconv_2)
        output = keras.layers.Conv2DTranspose(1, 11, strides=1, activation=keras.activations.tanh, padding="valid")(deconv_3)

        model = keras.Model(inputs=input, outputs=output)

        optimiser = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.91, beta_2=0.999)

        model.compile(optimizer=optimiser, loss=keras.losses.MSLE, metrics=["MeanSquaredLogarithmicError"])

        self.model = model
