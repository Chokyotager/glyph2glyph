from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
import json
import random

def trimAndCentre (im, box):

    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()

    im = im.crop(bbox)

    width, height = im.size

    offset = ((128 - width) // 2, (128 - height) // 2)
    bg.paste(im, offset)

    return bg

class Data ():

    def __init__ (self):

        data = json.load(open("data/min-trad.json"))
        kanji = list(data.keys())

        self.kanji = kanji

    def drawChar (self, char, font):

        assert isinstance(char, str)

        W, H = (128, 128)

        image = Image.new("L", (W, H), color=255)

        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(font, 128)

        w, h = draw.textsize(char, font=font)
        draw.text(((W - w)/2, (H - h)/2 - 10), char, fill=0, font=font)

        #image.save(data_dir + "out.png")

        image = trimAndCentre(image, box=(128, 128))

        return np.expand_dims(np.array(image), 2)

    def getCharPair (self, char):

        assert isinstance(char, str)

        rrpl = self.drawChar(char, font="font/RRPL.ttf")
        arial = self.drawChar(char, font="font/arial_unicode.ttf")

        return rrpl/127.5 - 1, arial/127.5 - 1

    def getCharPairs (self, batch=100):

        input = list()
        output = list()

        for i in range(batch):

            kanji = random.choice(self.kanji)
            rrpl, arial = self.getCharPair(kanji)

            input.append(rrpl)
            output.append(arial)

        return np.array(input), np.array(output)

    def drawOutput (self, array):

        array = np.uint8(127.5 * np.squeeze(array, 2) + 127.5)
        return Image.fromarray(array, "L")
