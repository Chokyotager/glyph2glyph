from data import Data

data = Data()
kanji = data.kanji

for i in range(len(kanji)):

    rrpl, arial = data.getCharPair(data.kanji[i])

    data.drawOutput(rrpl).save("images/rrpl/" + str(i) + ".png")
    data.drawOutput(arial).save("images/arial/" + str(i) + ".png")
