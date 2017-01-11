#!/usr/bin/env python

import time
import numpy as np
import os
import struct
import pymoos

class PyReadCBFFiles(object):

	def __init__(self):
		''' MOOSApp Attributes '''
		self.server_host = 'localhost'			#MOOSBD IP - eventually a .moos parser should be written for Python to do this stuff
		self.server_port = 9000					#MOOSDB port
		self.moos_app_name = 'PyReadCBFFiles'	#MOOSApp name
		self.time_warp = 1						#timewarp for simulation
		self.dir_path = '/home/rypkema/Workspace/Sandshark/logs_2016/sandshark_2016-07-26/sandshark_iMCC1608FS/track_trail_2/'
		self.pause_time = 1.0

		''' Initialzie Python-MOOS Communications '''
		self.comms = pymoos.comms()
		self.comms.set_on_connect_callback(self.on_connect)
		self.comms.set_on_mail_callback(self.on_mail)
		self.comms.run(self.server_host, self.server_port, self.moos_app_name)
		pymoos.set_moos_timewarp(self.time_warp)

	def run(self):
		''' Main loop simply reads files in directory, and publishes binary data to MOOSDB '''
		files = os.listdir(self.dir_path)
		natsort(files)
		for f in files:
			if f.endswith('.txt'):
				filepath = self.dir_path+f
				print filepath
				data = np.loadtxt(filepath, delimiter=',')
				binary_data = struct.pack('%sf' % len(data.ravel().tolist()), *data.ravel().tolist())
				self.comms.notify('DAQ_NUM_CHANNELS', data.shape[1], pymoos.time())
				self.comms.notify('DAQ_NUM_SAMPLES', data.shape[0], pymoos.time())
				self.comms.notify_binary('DAQ_BINARY_DATA', binary_data, pymoos.time())
				time.sleep(self.pause_time)
				
	def on_connect(self):
		''' On connection to MOOSDB, register for desired MOOS variables (allows for * regex) e.g. register('variable', 'community', 'interval')
		self.comms.register('NODE_*_PING','NODE_*',0) '''
		return True

	def on_mail(self):
		''' On receipt of new mail with MOOS variables we are interested in, parse the mail with possible accessors:
		msg.trace(), msg.time(), msg.name(), msg.key(), msg.is_name(), msg.source(), msg.is_double(), msg.double(), msg.double_aux(),
		msg.is_string(), msg.string(), msg.is_binary(), msg.binary_data(), msg.binary_data_size(), msg.mark_as_binary() '''
		return True

def try_int(s):
	"Convert to integer if possible."
	try: return int(s)
	except: return s

def natsort_key(s):
	"Used internally to get a tuple by which s is sorted."
	import re
	return map(try_int, re.findall(r'(\d+|\D+)', s))

def natcmp(a, b):
	"Natural string comparison, case sensitive."
	return cmp(natsort_key(a), natsort_key(b))

def natcasecmp(a, b):
	"Natural string comparison, ignores case."
	return natcmp(a.lower(), b.lower())

def natsort(seq, cmp=natcasecmp):
	"In-place natural string sort."
	seq.sort(cmp)

pyreadcbffile = PyReadCBFFiles()
pyreadcbffile.run()