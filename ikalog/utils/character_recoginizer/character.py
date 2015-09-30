#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  IkaLog
#  ======
#  Copyright (C) 2015 Takeshi HASEGAWA
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

import cv2
import numpy as np
import pickle

from ikalog.utils.character_recoginizer import *

class PerCharacter(object):

    def cut(self, img, img_hist_x):

        chars = []
        in_char = False
        x_start = None
        last_x = None
        for x in range(len(img_hist_x)):
            if in_char:
                if img_hist_x[x] > 0:
                    continue
                else:
                    chars.append((x_start, x - 1))
                    in_char = False
            else:
                if img_hist_x[x] > 0:
                    x_start = x
                    in_char = True
                else:
                    continue

        return chars

    def __init__(self):
        pass

array0to1280 = np.array(range(1280), dtype=np.int32)


class CharacterRecoginizer(object):

    def FES_NAME(self, img):
        # フェスの検出の場合は黄色文字を抽出。
        yellow = cv2.inRange(img_fes_title_hsv[:, :, 0], 32 - 2, 32 + 2)
        yellow2 = cv2.inRange(img_fes_title_hsv[:, :, 2], 240, 255)
        img_fes_title_mask = np.minimum(yellow, yellow2)
        return fes_yellow_mask

    def WHITE_STRING(self, img_hsv):
        # そうでなければ白だよね。
        white_mask_s = cv2.inRange(img_hsv[:, :, 1], 0, 48)
        white_mask_v = cv2.inRange(img_hsv[:, :, 2], 224, 256)
        white_mask = np.minimum(white_mask_s, white_mask_v)
        return white_mask

    def save_model_to_file(self, file):
        f = open(file, 'wb')
        pickle.dump([self.samples, self.responses], f)
        f.close()

    def load_model_from_file(self, file):
        f = open(file, 'rb')
        l = pickle.load(f)
        f.close()
        self.samples = l[0]
        self.responses = l[1]

    def add_sample(self, response, img):
        img = cv2.resize(
            img, (self.sample_width, self.sample_height), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sample = img.reshape((1, img.shape[0] * img.shape[1]))

        if self.samples is None:
            self.samples = np.empty((0, img.shape[0] * img.shape[1]))
            self.responses = []

        self.samples = np.append(self.samples, sample, 0)

        r = response
        try:
            response = ord('0') + int(response)
        except:
            response = ord(response)
            pass

        self.responses.append(response)

    def train(self):
        samples = np.array(self.samples, np.float32)
        responses = np.array(self.responses, np.float32)
        responses = responses.reshape((responses.size, 1))
        responses = np.array(self.responses, np.float32)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
        self.trained = True

    def extract_characters(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_chars = self.WHITE_STRING(img_hsv)

        # 生成されたデータが文字色が255, それ以外が 0 なので
        # 平均 16 をきっているようなら 1/16 以下ということに
        # isString = np.sum(img_fes_title_mask) > img_fes_title_mask.shape[
        #    0] * img_fes_title_mask.shape[1] * 16

        # 文字と判断したところを 1 にして縦に足し算
        img_chars1 = np.sum(img_chars / 255, axis=0)  # 列毎の検出dot数

        array0to1280 = np.array(range(1280), dtype=np.int32)
        #img_chars1_hist_x = np.extract(img_chars1 > 0, array0to1280[0:len(img_chars1) -1])

        # FixMe: 255 px まで
        img_chars1_hist_x = np.minimum(
            img_chars1, array0to1280[0:len(img_chars1)])

        # ToDo: フェスタイトルのように長さが決まっている場合は固定長で切り出す
        # 一文字単位で認識する場合はヒストグラムから文字の位置リストを作る
        x_cutter = PerCharacter()

        char_tuples = x_cutter.cut(img_chars, img_chars1_hist_x)

        characters = []
        img_chars = np.sum(img_chars[:, :], axis=1)  # 行毎の検出dot数
        img_char_extract_y = np.extract(
            img_chars > 0, array0to1280[0:len(img_chars)])
        y1 = np.amin(img_char_extract_y)
        y2 = np.amax(img_char_extract_y) + 1
        for t in char_tuples:

            img_char_final = img[y1:y2, t[0]: t[1]]
            characters.append(img_char_final)

        return characters

    def find_samples(self, img):
        characters = self.extract_characters(img)
        samples = []
        for img in characters:
            img_scaled = cv2.resize(
                img, (self.sample_width, self.sample_height), interpolation=cv2.INTER_NEAREST)
            samples.append(img)
        return samples

    def match1(self, img):
        if (img.shape[0] != self.sample_width) or (img.shape[1] != self.sample_height):
            img = cv2.resize(
                img, (self.sample_width, self.sample_height), interpolation=cv2.INTER_NEAREST)

        if (len(img.shape) > 2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        raito = np.sum(
            img) / (img.shape[0] * img.shape[1]) if np.sum(img) != 0 else 0.0

        if raito < 0.1:
            # ほぼ真っ黒
            return 0

        sample = img.reshape((1, img.shape[0] * img.shape[1]))
        sample = np.array(sample, np.float32)

        k = 3

        retval, results, neigh_resp, dists = self.model.findNearest(sample, k)

        d = chr(int(results.ravel()))
        return d

    def match(self, img):
        if not self.trained:
            return None

        samples = self.find_samples(img)

        s = ''
        for sample in samples:
            c = self.match1(sample)
            s = s + c

        return s

    def match_digits(self, img):
        return int(self.match(img))

    def __init__(self):
        self.trained = False

        self.sample_width = 13
        self.sample_height = 17

        self.samples = None  # np.empty((0, 21 * 14))
        self.responses = []
        self.model = cv2.ml.KNearest_create()
