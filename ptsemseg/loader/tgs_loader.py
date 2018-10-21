import os
import random
import cv2
import torch
import pandas as pd
import numpy as np

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


class tgsLoader(data.Dataset):

    mean_rgb = {'pascal': [103.939, 116.779, 123.68], 'tgs': [123.03]}

    def __init__(self, root, split="train", is_transform=True,
                 img_size=(101, 101), augmentations=None, img_norm=True,
                 version='tgs', no_gt=False, num_k_split=0, max_k_split=0, sd=1234, r_pad=0):

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.no_gt = no_gt
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.r_pad = r_pad
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, 'images')
        self.annotations_base = os.path.join(self.root, self.split, 'masks')

        fs = recursive_glob(rootdir=self.images_base, suffix='.png')
        if max_k_split == 0 or num_k_split == 0 or split == 'test': # Select all files in the split
            self.files[split] = fs
        else: # Select the k-th fold files in the split
            torch.manual_seed(sd)
            rp = torch.randperm(len(fs))
            start_f = len(fs) // max_k_split * (num_k_split-1)
            end_f = len(fs) // max_k_split * num_k_split
            self.ind = rp[start_f:end_f]
            if split == 'train':
                self.files[split] = [f for i, f in enumerate(fs) if i not in self.ind]
            else:#if split == 'val':
                self.files[split] = [f for i, f in enumerate(fs) if i in self.ind]

        self.depth_df = pd.read_csv(os.path.join(self.root, 'depths.csv'), index_col=0)

        self.valid_classes = [0, 255]
        self.ignore_index = 250
        self.lbl_thr = 0

        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        self.label_colours = dict(zip(range(self.n_classes), self.valid_classes))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype=np.uint8)

        if self.no_gt:
            lbl = [os.path.basename(img_path)]
        else:
            lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        lbl_dp = self.depth_df.loc[os.path.basename(img_path).split('.')[0], 'z']
        lbl_dp = torch.FloatTensor([lbl_dp])
        name = [os.path.basename(img_path)]

        if not self.no_gt and self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, lbl_dp, name

    def transform(self, img, lbl, pad_mode='symmetric'):
        if img.shape[0] != self.img_size[0] or img.shape[1] != self.img_size[1]:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR) # cv2.resize shape: (W, H)

        if len(img.shape) == 3:
            img = img[:, :, ::-1] # RGB -> BGR

        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0 # Rescale images from [0, 255] to [0, 1]

        if self.r_pad > 0:
            pad_width = ((self.r_pad,self.r_pad),(self.r_pad,self.r_pad),(0,0)) if len(img.shape) == 3 else ((self.r_pad,self.r_pad),(self.r_pad,self.r_pad))
            img = np.pad(img, pad_width, pad_mode)

        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1) # NHWC -> NCHW
        else:
            img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(img).float()

        if not self.no_gt:
            if lbl.shape[0] != self.img_size[0] or lbl.shape[1] != self.img_size[1]:
                lbl = cv2.resize(lbl, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST) # cv2.resize shape: (W, H)

            if 'train' in self.split and lbl.sum() <= self.lbl_thr:
                lbl[lbl == 1] = self.ignore_index

            if self.r_pad > 0:
                lbl = np.pad(lbl, ((self.r_pad,self.r_pad),(self.r_pad,self.r_pad)), pad_mode)

            lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp, img_norm=False):
        out = np.zeros((temp.shape[0], temp.shape[1]))
        for l in range(self.n_classes):
            out[temp == l] = self.label_colours[l]

        out = out / 255.0 if img_norm else out
        return out

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


    def RLenc(self, img, order='F', format=True):
        """
        https://www.kaggle.com/bguberfain/unet-with-depth

        img is binary mask image, shape (r,c)
        order is down-then-right, i.e. Fortran
        format determines if the order needs to be preformatted (according to submission rules) or not

        returns run length as an array or string (if format is True)
        """
        bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
        runs = []  ## list of run lengths
        r = 0  ## the current run length
        pos = 1  ## count starts from 1 per WK
        for c in bytes:
            if c == 0:
                if r != 0:
                    runs.append((pos, r))
                    pos += r
                    r = 0
                pos += 1
            else:
                r += 1

        # if last run is unsaved (i.e. data ends with 1)
        if r != 0:
            runs.append((pos, r))
            pos += r
            r = 0

        if format:
            z = ''
            for rr in runs:
                z += '{} {} '.format(rr[0], rr[1])
            return z[:-1]
        else:
            return runs

    def decode_RLenc(self, rle_mask, order='F'):
        img = np.zeros(self.img_size[0] * self.img_size[1])
        if isinstance(rle_mask, str): # py2: basestring, py3: str
            for i, v in enumerate(rle_mask.split()):
                if i % 2 == 0:
                    pos = int(v)-1
                else:
                    r = int(v)
                    img[pos:pos+r] = 1
        return self.decode_segmap(img.reshape(self.img_size[0], self.img_size[1], order=order))
