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
from __future__ import print_function
from .ikautils import IkaUtils
from .matcher import IkaMatcher
from .glyph_recoginizer import IkaGlyphRecoginizer
from .weapon_recoginizer.nin import NIN
from .weapon_recoginizer.alex import Alex
from .weapon_recoginizer.alexbn import AlexBN
from .weapon_recoginizer.googlenet import GoogLeNet
from .weapon_recoginizer.googlenetbn import GoogLeNetBN
from .weapon_recoginizer.weapon_recoginizer import IkaWeaponRecoginizer
from .character_recoginizer import CharacterRecoginizer
from .character_recoginizer.number import NumberRecoginizer
from .character_recoginizer.udemae import UdemaeRecoginizer
from .character_recoginizer.fes_gender import FesGenderRecoginizer
from .character_recoginizer.fes_level import FesLevelRecoginizer
