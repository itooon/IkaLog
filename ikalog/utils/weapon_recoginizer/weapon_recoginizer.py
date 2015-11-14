#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  IkaLog
#  ======
#  Copyright (C) 2015 Takeshi HASEGAWA
#  Copyright (C) 2015 Hiromochi Itoh
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import print_function

from ikalog.utils.weapon_recoginizer import *

import sys
sys.modules['nin'] = nin
sys.modules['alex'] = alex
sys.modules['alexbn'] = alexbn
sys.modules['googlenet'] = googlenet
sys.modules['googlenetbn'] = googlenetbn

import cv2
import numpy as np

import pickle

from chainer import cuda
from chainer import optimizers

class IkaWeaponRecoginizer(object):

    # read the image. (for weapons)
    #
    # @param img    the source image
    # @param debug   enable debug function
    # @return image the result
    def normalize_weapon_image(self, img, debug=False):
        image = np.asarray(cv2.resize(img, (256,256))).transpose(2, 0, 1)

        if debug:
            cv2.imshow('orig', cv2.resize(img, (160,160)))
            ch = 0xFF & cv2.waitKey(1)
            if ch == ord('q'):
               sys.exit()

        top = left = self.cropwidth / 2
        bottom = self.model.insize + top
        right = self.model.insize + left
        image = image[:, top:bottom, left:right].astype(np.float32)
        image -= self.mean_image[:, top:bottom, left:right]
        image /= 255
        return image

    def match(self, img):
        if not self.trained:
            return None, None

        img = self.normalize_weapon_image(img, debug=False)
        x = np.ndarray(
                (1, 3, self.model.insize, self.model.insize), dtype=np.float32)
        x[0]=img
        x = self.xp.asarray(x)
        score = self.model.predict(x)

        prediction = list(zip(score.data[0].tolist(), self.categories))
        prediction.sort(key=lambda x: x[0] , reverse=True)
        (score, name) = prediction[0]

        return name, score

    def load_model(self, mean_path='data/mean.npy', categories_path='data/labels.txt', model_path='data/model', gpu=-1):
        self.mean_image = pickle.load(open(mean_path, 'rb'))
        self.categories = np.loadtxt(categories_path,
                              delimiter="\t",
                              dtype = str,
                              converters = {k:np.compat.asstr for k in range(1)})
        self.model = pickle.load(open(model_path,'rb'))

        if gpu >= 0:
            cuda.check_cuda_available()
            self.xp = cuda.cupy
            cuda.get_device(gpu).use()
            self.model.to_gpu()
        else:
            self.xp = np
            self.model.to_cpu()
        self.cropwidth = 256 - self.model.insize
        self.trained = True

    def __init__(self):
        self.trained = False

if __name__ == "__main__":
    import argparse
    import os
    import shutil
    from ikalog.utils import *

    parser = argparse.ArgumentParser(
    description='Weapon image inspection using chainer')
    parser.add_argument('image', help='Path to dir that include inspection image files')
    parser.add_argument('--model','-m',default='model', help='Path to model file')
    parser.add_argument('--mean', default='mean.npy',
                        help='Path to the mean file (computed by compute_mean.py)')
    parser.add_argument('--labels','-l',default='labels.txt', help='Path to labels file')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--output', '-o', help='image directory for categoriesing by dnn')
    args = parser.parse_args()
    test_weapon = IkaWeaponRecoginizer()
    test_weapon.load_model(args.mean, args.labels, args.model, args.gpu)
    try:
        output_enable = False
        args.output
    except NameError:
        pass
    else:
        output_enable = True
        os.makedirs(args.output, exist_ok=True)
    counter = 0
    fail_counter = 0
    for root, dirs, files in os.walk(args.image):
        for file in files:
            full_path = os.path.join(root,file)
            img = cv2.imread(full_path)
            if img is None:
                print("Failed to load image file.")
                print(full_path)
                fail_counter = fail_counter + 1
                continue
            height, width, channels = img.shape[:3]
            if height == 0:
                fail_counter = fail_counter + 1
                continue
            (result_name, result_accuracy) = test_weapon.match(img)
            counter = counter + 1
            dir_name = os.path.basename(root)
            if dir_name != result_name :
                fail_counter = fail_counter + 1
                print(dir_name)
                print(result_name)
            if output_enable:
                output_path = os.path.join(args.output, result_name)
                os.makedirs(output_path, exist_ok=True)
                output_path = os.path.join(output_path, file)
                shutil.copyfile(full_path, output_path)
    print(counter, fail_counter)
