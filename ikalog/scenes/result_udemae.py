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
import sys

import cv2

from ikalog.utils import *


class ResultUdemae(object):

    last_matched = False

    def match1(self, context):
        frame = context['engine']['frame']
#        gray_frame = cv2.cvtColor(context['engine']['frame'], cv2.COLOR_BGR2GRAY)
        matched = self.mask_udemae_msg.match(frame)
        return matched

    def match(self, context):
        if not self.match1(context):
            if ('result_udemae' in context['scene']) and ('udemae_str' in context['scene']['result_udemae']):
                # ウデマエが表示されていたのが消えた
                print(context['scene']['result_udemae'])

                callPlugins = context['engine']['service']['callPlugins']
                callPlugins('on_result_udemae')

                context['scene']['result_udemae'] = {}

            return False
        try:
            img_udemae = context['engine']['frame'][357:357+108, 450:450+190]
            cv2.imshow('udemae', img_udemae)
            img_udemae_num = context['engine']['frame'][310:310 + 185, 770:770+110]
            cv2.imshow('udemae_num', img_udemae_num)

            udemae_str = self.udemae_recoginizer.match(img_udemae)
            udemae_num = self.number_recoginizer.match(img_udemae_num)
        except:
            return False

        if not ('result_udemae' in context['scene']):
            context['scene']['result_udemae'] = {
                'udemae_str_pre': udemae_str,
                'udemae_num_pre': udemae_num,
            }
        context['scene']['result_udemae']['last_update'] = context['engine']['msec']
        context['scene']['result_udemae']['udemae_str'] = udemae_str
        context['scene']['result_udemae']['udemae_num'] = udemae_num

        return True

    def analyze(self, context):
        img_udemae = context['engine']['frame'][357:357+108, 450:450+190]
        cv2.imshow('udemae', img_udemae)
        img_udemae_num = context['engine']['frame'][310:310 + 185, 770:770+110]
        cv2.imshow('udemae_num', img_udemae_num)

        print(self.udemae_recoginizer.match(img_udemae))
        print(self.number_recoginizer.match(img_udemae_num))


    def __init__(self, debug=False):
        self.udemae_recoginizer = UdemaeRecoginizer()
        self.number_recoginizer = NumberRecoginizer()
        # "ウデマエ" 文字列。オレンジ色。 IkaMatcher の拡張が必要
        self.mask_udemae_msg = IkaMatcher(
            561, 245, 144, 52,
            img_file='masks/ui_udemae.png',
            threshold=0.5,
            orig_threshold=0.250,
            false_positive_method=IkaMatcher.FP_BACK_IS_BLACK,
            pre_threshold_value=205,
            label='result_udemae/Udemae',
            debug=debug,
        )

if __name__ == "__main__":
    target = cv2.imread(sys.argv[1])
    obj = ResultUdemae(debug=True)

    context = {
        'engine': {'frame': target},
        'game': {},
    }

    matched = obj.match(context)
    analyzed= obj.analyze(context)
    print("matched %s" % (matched))

    cv2.waitKey()
