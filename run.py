import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model import Model
from data import Data

data = Data()
model = Model().model

for i in range(10):

    print("Running iteration {}".format(i + 1))

    input, output = data.getCharPairs(100)
    model.fit(x=input, y=output)

# Test model
rrpl, arial = data.getCharPair("義")

predicted_kanji = model.predict(np.expand_dims(rrpl, 0))
predicted_kanji = np.squeeze(predicted_kanji, 0)

data.drawOutput(predicted_kanji).save("output.png")
