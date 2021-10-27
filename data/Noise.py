import numpy as np
from PIL import Image, ImageFilter


class SaltPepperNoise(object):
    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        noiseDensity = self.density
        sourceDensity = 1 - noiseDensity
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[noiseDensity / 2.0, noiseDensity / 2.0, sourceDensity])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class ReverseNoise(object):
    def __init__(self, probability=0):
        self.probability = probability

    def __call__(self, img):
        img = np.array(img)
        probability = self.probability
        flag = np.random.choice((0, 1), p=[1 - probability, probability])
        if flag:
            img = 255 - img
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class GaussBlur(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img

class NormImage(object):
    def __init__(self,probability=1):
        self.probability = probability
    def __call__(self, img):
        img = np.array(img)
        probability = self.probability
        flag = np.random.choice((0,1), p=[1 - probability, probability])
        if flag:
            img = np.floor(256*(img-img.min()+1)/(img.max()-img.min()+1)-1)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

