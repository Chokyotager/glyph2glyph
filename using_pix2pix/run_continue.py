import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pix2pix import Pix2Pix
from data import Data

data = Data()
pix2pix = Pix2Pix()

batch_size = 6

accuracy = 0
ignore_runs = 0

pix2pix.generator.load_weights("saves/save_NEW2_X2.h5")
pix2pix.discriminator.load_weights("saves/save_NEW2_X2.discriminator.h5")
pix2pix.combined.load_weights("saves/save_NEW2_X2.combined.h5")

# Train the model
for i in range(1000000):

    if i % 20 == 0:

        rrpl, arial = data.getCharPair("龍")
        predicted = pix2pix.generator.predict(np.expand_dims(rrpl, 0))

        data.drawOutput(predicted[0]).save("output.png")

    # Adversarial loss ground truths
    valid = np.ones((batch_size,))
    fake = np.zeros((batch_size,))

    rrpl, arial = data.getCharPairs(batch_size)
    fake_B = pix2pix.generator.predict(rrpl)

    if accuracy > 1 and ignore_runs < 0:
        ignore_runs = 3

    if ignore_runs <= 0:

        combined_A = np.concatenate([rrpl, rrpl])
        combined_B = np.concatenate([arial, fake_B])
        combined_result = np.concatenate([valid, fake])

        d_loss = pix2pix.discriminator.train_on_batch([combined_A, combined_B], combined_result)

    g_loss = pix2pix.combined.train_on_batch([rrpl, arial], [valid, arial])
    ignore_runs -= 1

    accuracy = d_loss[1]

    print("[Iteration {}] [D loss: {}, acc: {}] [G loss: {}]".format(i,
            d_loss[0], accuracy,
            g_loss[0]))

    if i % 50 == 0:
        print("SAVING...")
        pix2pix.generator.save("saves/save_NEW2_X2.h5")
        pix2pix.combined.save("saves/save_NEW2_X2.combined.h5")
        pix2pix.discriminator.save("saves/save_NEW2_X2.discriminator.h5")
