import numpy as np
import torchvision.transforms.functional


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return torchvision.transforms.functional.pad(image, padding, 0, "constant")
