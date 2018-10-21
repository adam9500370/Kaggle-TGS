# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

class Compose(object):
    def __init__(self, augmentations, is_random_aug=False):
        self.augmentations = augmentations
        self.is_random_aug = is_random_aug

    def __call__(self, img, mask):
        img = Image.fromarray(img, mode='RGB') if len(img.shape) == 3 else Image.fromarray(img, mode='L')
        mask = Image.fromarray(mask, mode='L')
        assert img.size == mask.size

        for a in self.augmentations:
            if not self.is_random_aug or (self.is_random_aug and random.random() < 0.5):
                img, mask = a(img, mask)

        return np.array(img), np.array(mask, dtype=np.uint8)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
        assert img.size == mask.size

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size

        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask

        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)

        return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size, change_ar=True, min_area=0.45):
        self.size = size
        self.change_ar = change_ar
        self.min_area = min_area

    def __call__(self, img, mask):
        assert img.size == mask.size

        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2.0) if self.change_ar else 1.0

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))

                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size), Image.NEAREST)

        if self.size == img.size[0] and self.size == img.size[1]:
            return img, mask

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])
        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        return self.crop(*self.scale(img, mask))


class RandomTranslateWithReflect(object):
    #https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/mean_teacher/data.py
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, img, mask):
        assert img.size == mask.size

        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = img.size

        flipped_lr = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = img.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = img.transpose(Image.ROTATE_180)

        new_image = Image.new(img.mode, (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(img, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        flipped_lr = mask.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = mask.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = mask.transpose(Image.ROTATE_180)

        new_mask = Image.new(mask.mode, (xsize + 2 * xpad, ysize + 2 * ypad))

        new_mask.paste(mask, (xpad, ypad))

        new_mask.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_mask.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_mask.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_mask.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_mask.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_mask.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_mask.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_mask.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_mask = new_mask.crop((xpad - xtranslation,
                                  ypad - ytranslation,
                                  xpad + xsize - xtranslation,
                                  ypad + ysize - ytranslation))

        return new_image, new_mask
