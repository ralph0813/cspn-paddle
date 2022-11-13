import numpy as np
from paddle.vision import BaseTransform
import paddle.vision.transforms.functional as F


class Rotation(BaseTransform):
    def __init__(
            self,
            degree,
            interpolation='nearest',
            expand=False,
            center=None,
            fill=0,
            keys=None,
    ):
        super(Rotation, self).__init__(keys)
        assert isinstance(degree, int) or isinstance(degree, float)

        self.degree = degree if degree > 0 else degree + 360
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def _apply_image(self, img):
        angle = self.degree
        img = np.array(img)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return F.rotate(img, angle, self.interpolation, self.expand, self.center, self.fill)
