#!/usr/bin/env python

import numpy as np
import scipy as sp 	# FASTEST FFT/IFFT!
import math
import itertools
import time

from czt import CZT
from cbf_mac_alt import cbf_mac 	#cbf_mac_alt takes ifft of cbf and finds max value in time (max over all time)
#from cbf_mac import cbf_mac 		#cbf mac sums over all frequency components and returns that value (average over all frequencies)

class CBF(object):

	def __init__(self, phone_array, replica, freq_sampling, freq_lower, freq_upper, nfft, nsamples):
		self.pos = phone_array
		self.num_pos = self.pos.shape[0]
		self.replica = replica
		self.freq_sampling = freq_sampling
		self.freq_lower = freq_lower
		self.freq_upper = freq_upper
		self.nfft = nfft
		self.nsamples = nsamples
		self.init_divisions = 50
		self.succ_divisions = 20
		self.init_swath = 10.0*np.pi/180.0
		self.end_swath = self.succ_divisions*np.pi/180.0
		self.factor_swath = 0.5
		self.sound_speed = 1300.0
		self.matched_filter_offset = -300.0
		self.init_look_elevations = np.linspace(1,180,self.init_divisions)*np.pi/180.0
		self.init_look_azimuths = np.linspace(1,360,self.init_divisions*2)*np.pi/180.0
		self.init_cbf_filter = self.get_cbf_filter(self.pos, self.sound_speed, self.init_look_azimuths, self.init_look_elevations, self.nfft, self.freq_lower, self.freq_upper)
		self.matched_filter = self.get_matched_filter(self.replica, self.num_pos, self.freq_sampling, self.nfft, self.freq_lower, self.freq_upper)
		self.full_matched_filter = self.get_full_matched_filter(self.replica, self.num_pos, self.nsamples)
		self.czt_filter = self.get_czt_filter(self.freq_sampling, self.nfft, self.freq_lower, self.freq_upper, self.nsamples)
		self.cbf_output_init_heatmap = None
		self.cbf_output_max_value = None
		self.cbf_output_max_azimuth = None
		self.cbf_output_max_elevation = None
		self.cbf_output_max_range = None
		self.cbf_output_var_range = None

	def run(self, data):
		# timestart = time.time()
		# Fdata = np.fft.fft(data, self.nsamples, axis=0)
		# matched_data = np.fft.ifft(Fdata*self.full_matched_filter, axis=0)
		Fdata = sp.fftpack.fft(data, self.nsamples, axis=0)
		matched_data = sp.fftpack.ifft(Fdata*self.full_matched_filter, axis=0)
		max_range_idxs = np.argmax(np.abs(matched_data), axis=0)
		max_ranges = (max_range_idxs+self.matched_filter_offset)/self.freq_sampling*self.sound_speed
		self.cbf_output_max_range = np.mean(max_ranges)
		self.cbf_output_var_range = np.var(max_ranges)
		
		### OPTION TO CHECK VARIANCE ON 3 OF 4 ELEMENTS AND BEAMFORM WITH ONLY THOSE
		# var_4 = np.var(max_ranges)
		# var_idxs = np.array([0,1,2,3])
		# self.init_cbf_filter = self.get_cbf_filter(self.pos, self.sound_speed, self.init_look_azimuths, self.init_look_elevations, self.nfft, self.freq_lower, self.freq_upper)
		# if var_4 > 10:
		# 	min_var_3 = 100
		# 	for comb in itertools.combinations(enumerate(max_ranges),3):
		# 		comb = np.array(comb)
		# 		var_3 = np.var(comb[:,1])
		# 		if var_3 <= min_var_3:
		# 			min_var_3 = var_3
		# 			var_idxs = comb[:,0]
		# self.init_cbf_filter = self.get_cbf_filter(self.pos[var_idxs.astype(int),:], self.sound_speed, self.init_look_azimuths, self.init_look_elevations, self.nfft, self.freq_lower, self.freq_upper)
		# self.matched_filter = self.get_matched_filter(self.replica, len(var_idxs), self.freq_sampling, self.nfft, self.freq_lower, self.freq_upper)
		# self.cbf_output_init_heatmap = self.beamform(self.init_look_azimuths, self.init_look_elevations, data[:,var_idxs.astype(int)], self.init_cbf_filter, self.matched_filter, self.czt_filter, self.nfft, len(var_idxs))
		
		self.cbf_output_init_heatmap = self.beamform(self.init_look_azimuths, self.init_look_elevations, data, self.init_cbf_filter, self.matched_filter, self.czt_filter, self.nfft, self.num_pos)
		self.cbf_output_max_value = np.max(self.cbf_output_init_heatmap)
		row_idx = np.argmax(np.max(self.cbf_output_init_heatmap, axis=1))
		col_idx = np.argmax(np.max(self.cbf_output_init_heatmap, axis=0))
		self.cbf_output_max_elevation = self.init_look_elevations[row_idx]
		self.cbf_output_max_azimuth = self.init_look_azimuths[col_idx]

		## OPTION TO SUCCESSIVELY BEAMFORM WITH SMALLER SWATHS CENTERED AROUND PREVIOUSLY BEAMFORMING SOLUTION
		swath = self.init_swath
		while True:
			lower_elev = self.cbf_output_max_elevation - swath/2
			upper_elev = self.cbf_output_max_elevation + swath/2
			lower_azim = self.cbf_output_max_azimuth - swath/2
			upper_azim = self.cbf_output_max_azimuth + swath/2
			if (lower_elev < np.pi/180):
				lower_elev = np.pi/180
			if (upper_elev > np.pi):
				upper_elev = np.pi
			if (lower_azim < np.pi/180):
				lower_azim = np.pi/180
			if (upper_azim > 2*np.pi):
				upper_azim = 2*np.pi
			look_elevations = np.linspace(lower_elev,upper_elev,self.succ_divisions)
			look_azimuths = np.linspace(lower_azim,upper_azim,self.succ_divisions)
			cbf_filter = self.get_cbf_filter(self.pos, self.sound_speed, look_azimuths, look_elevations, self.nfft, self.freq_lower, self.freq_upper)
			cbf_output = self.beamform(look_azimuths, look_elevations, data, cbf_filter, self.matched_filter, self.czt_filter, self.nfft, self.num_pos)
			self.cbf_output_max_value = np.max(cbf_output)
			row_idx = np.argmax(np.max(cbf_output, axis=1))
			col_idx = np.argmax(np.max(cbf_output, axis=0))
			self.cbf_output_max_elevation = look_elevations[row_idx]
			self.cbf_output_max_azimuth = look_azimuths[col_idx]
			if swath < self.end_swath:
				break
			swath = swath*self.factor_swath
		# print time.time() - timestart, 'multiply option 1'

	def get_cbf_filter(self, phones, sound_speed, look_azimuths, look_elevations, nfft, f1, f2):
		cbf_filter = np.zeros([len(look_elevations),len(look_azimuths),nfft,phones.shape[0]], order='C')*0j
		fft_freqs = np.array([np.linspace(f1,f2,nfft)])
		time_delays = np.zeros([phones.shape[0],1])
		a = np.zeros([3,1])
		for i in range(0,len(look_elevations)):
			for j in range(0,len(look_azimuths)):
				a[0,0] = -math.sin(look_elevations[i])*math.cos(look_azimuths[j])
				a[1,0] = -math.sin(look_elevations[i])*math.sin(look_azimuths[j])
				a[2,0] = -math.cos(look_elevations[i])
				time_delays = np.dot(phones,a)/sound_speed
				cbf_filter[i,j,:,:] = np.transpose(np.conj(np.exp(-2*1j*np.pi*np.dot(time_delays,fft_freqs))))
		return cbf_filter

	def get_matched_filter(self, replica, num_phones, sampling_rate, nfft, f1, f2):
		w = np.exp(-1j*2*np.pi*(f2-f1)/(nfft*sampling_rate))
		a = np.exp(1j*2*np.pi*f1/sampling_rate)
		c = CZT(replica.shape[1],nfft,w,a)
		matched_filter = np.conj(c(replica))
		return matched_filter[0,:]

	def get_full_matched_filter(self, replica, num_phones, nfft):
		# Freplica = np.conj(np.fft.fft(replica, nfft))
		Freplica = np.conj(sp.fftpack.fft(replica, nfft))
		full_matched_filter = np.tile(np.transpose(Freplica), (1,num_phones))
		return full_matched_filter

	def next_greater_power_of_2(self, x):
		return 2**(x-1).bit_length()

	def get_czt_filter(self, sampling_rate, nfft, f1, f2, filter_size):
		w = np.exp(-1j*2*np.pi*(f2-f1)/(nfft*sampling_rate))
		a = np.exp(1j*2*np.pi*f1/sampling_rate)
		czt_filter = CZT(filter_size,nfft,w,a)
		return czt_filter

	def beamform(self, look_azimuths, look_elevations, phone_data, cbf_filter, matched_filter, czt_filter, nfft, num_phones):
		cbf_data = np.zeros([len(look_elevations),len(look_azimuths)])
		Fphone_data = np.transpose(czt_filter(np.transpose(phone_data)))
		cbf_mac(len(look_elevations), len(look_azimuths), nfft, num_phones, Fphone_data, cbf_filter, matched_filter, cbf_data)
		return cbf_data
