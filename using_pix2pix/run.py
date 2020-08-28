import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pix2pix import Pix2Pix
from data import Data

data = Data()
pix2pix = Pix2Pix()

batch_size = 10

# Train the model
for i in range(100):

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + pix2pix.disc_patch)
    fake = np.zeros((batch_size,) + pix2pix.disc_patch)

    rrpl, arial = data.getCharPairs(batch_size)

    fake_A = pix2pix.generator.predict(arial)

    d_loss_real = pix2pix.discriminator.train_on_batch([rrpl, arial], valid)
    d_loss_fake = pix2pix.discriminator.train_on_batch([fake_A, arial], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    g_loss = pix2pix.combined.train_on_batch([rrpl, arial], [valid, rrpl])

    print(d_loss)
    print(g_loss)
