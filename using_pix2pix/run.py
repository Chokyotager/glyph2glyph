import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pix2pix import Pix2Pix
from data import Data

data = Data()
pix2pix = Pix2Pix()

# Train the model
