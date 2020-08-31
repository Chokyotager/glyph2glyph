from data import Data
import numpy as np

data = Data()
rrpl, arial = data.getCharPair("正")

result = arial.max()

print(result)
data.drawOutput(rrpl).save("output.png")
