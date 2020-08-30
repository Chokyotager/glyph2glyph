from data import Data
import numpy as np

data = Data()
rrpl, arial = data.getCharPair("正")

result = arial.min()

print(rrpl.shape)
