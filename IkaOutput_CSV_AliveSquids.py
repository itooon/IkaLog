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

#
#  This module is still in proof of concept, and subject to change.
#

from IkaUtils import *
from datetime import datetime
import time

## IkaLog Output Plugin: Write 'Alive Squids' CSV data
#
class IkaOutput_CSV_AliveSquids:

	##
	# Write a line to text file.
	# @param self     The Object Pointer.
	# @param record   Record (text)
	#
	def writeRecord(self, file, record):
		try:
			csv_file = open(file, "a")
			csv_file.write(record)
			csv_file.close
		except:
			print("CSV: Failed to write CSV File")

	##
	# onGameIndividualResult Hook
	# @param self      The Object Pointer
	# @param context   IkaLog context
	#
	def onGameIndividualResult(self, context):
		time = 0
		csv = ["tick,y\n", "tick,y\n"]

		for sample in context['game']['livesTrack']:
			time = time + 1
			num_team = 0
			for team in sample:
				num_squid = 0
				for alive in team:
					num_squid = num_squid + 1

					if alive:					
						csv[num_team] = "%s%d, %d\n" % (csv[num_team], time, num_squid)
				num_team = num_team + 1

		num_team = 0

		t = datetime.now()
		t_str = t.strftime("%Y%m%d_%H%M")

		for f in csv:
			self.writeRecord('alivesquids_%s_team%d.csv' % (t_str, num_team), f)
			num_team = num_team + 1
