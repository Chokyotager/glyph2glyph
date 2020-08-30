import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pix2pix import Pix2Pix
from data import Data

data = Data()
pix2pix = Pix2Pix()

batch_size = 4

accuracy = 0
ignore_runs = 0

# Train the model
for i in range(10000):

    rrpl, arial = data.getCharPair("正")
    predicted = pix2pix.generator.predict(np.expand_dims(rrpl, 0))

    data.drawOutput(predicted[0]).save("output.png")

    # Adversarial loss ground truths
    valid = np.ones((batch_size,))
    fake = np.zeros((batch_size,))

    rrpl, arial = data.getCharPairs(batch_size)
    fake_A = pix2pix.generator.predict(rrpl)

    if accuracy > 1 and ignore_runs < 0:
        ignore_runs = 3

    if ignore_runs <= 0:

        combined_A = np.concatenate([rrpl, arial])
        combined_B = np.concatenate([rrpl, fake_A])
        combined_result = np.concatenate([valid, fake])

        d_loss = pix2pix.discriminator.train_on_batch([combined_A, combined_B], combined_result)

    g_loss = pix2pix.combined.train_on_batch([rrpl, arial], [valid, rrpl])
    ignore_runs -= 1

    accuracy = d_loss[1]

    print("[Iteration {}] [D loss: {}, acc: {}] [G loss: {}]".format(i,
            d_loss[0], accuracy,
            g_loss[0]))
