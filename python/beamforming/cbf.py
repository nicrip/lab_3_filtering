#!/usr/bin/env python

import numpy as np
import scipy as sp  # FASTEST FFT/IFFT!
import math
import itertools
import time

from czt import CZT
from cbf_mac_alt_mp import cbf_mac 	#cbf_mac_alt takes ifft of cbf and finds max value in time (max over all time)
#from cbf_mac_alt import cbf_mac 	#non-multithreaded
#from cbf_mac_mp import cbf_mac 	#cbf mac sums over all frequency components and returns that value (average over all frequencies)
#from cbf_mac import cbf_mac 		#non-multithreaded

class CBF(object):

	def __init__(self, phone_array, replica, freq_sampling, freq_lower, freq_upper, nfft, nsamples):
		''' Internal Beamformer Attributes '''
		self.divisions_elevation = 50
		self.divisions_azimuth = 100
		self.sound_speed = 1300.0				#(aka matched filter multiplier)
		self.matched_filter_offset = -300.0
		self.slice_data_at_MF = True
		self.slice_data_at_MF_size = 400

		''' Passed Arguments Beamformer Attributes '''
		self.pos = phone_array
		self.num_pos = self.pos.shape[0]
		self.replica = replica
		self.freq_sampling = freq_sampling
		self.freq_lower = freq_lower
		self.freq_upper = freq_upper
		self.nfft = nfft
		self.nsamples = nsamples
		
		''' Initialize Beamformer Objects '''
		self.look_elevations = np.linspace(1,180,self.divisions_elevation)*np.pi/180.0
		self.look_azimuths = np.linspace(1,360,self.divisions_azimuth)*np.pi/180.0
		self.cbf_filter = self.get_cbf_filter(self.pos, self.sound_speed, self.look_azimuths, self.look_elevations, self.nfft, self.freq_lower, self.freq_upper)
		self.matched_filter = self.get_matched_filter(self.replica, self.num_pos, self.freq_sampling, self.nfft, self.freq_lower, self.freq_upper)
		self.full_matched_filter = self.get_full_matched_filter(self.replica, self.num_pos, self.nsamples)
		self.czt_filter = self.get_czt_filter(self.freq_sampling, self.nfft, self.freq_lower, self.freq_upper, self.nsamples)
		self.cbf_output_heatmap = np.zeros((self.divisions_elevation, self.divisions_azimuth))
		self.cbf_output_max_value = 0.0
		self.cbf_output_max_azimuth = 0.0
		self.cbf_output_max_elevation = 0.0
		self.cbf_output_max_range = 0.0
		self.cbf_output_var_range = 0.0
		self.temp_data_matched_filter = np.zeros((self.nsamples,self.num_pos), order='C', dtype=np.complex128)
		self.temp_data_cbf_filter = np.zeros((self.nfft,self.num_pos), order='C', dtype=np.complex128)

	def run(self, data):
		timestart = time.time()
		self.temp_data_matched_filter[:] = sp.fftpack.fft(data, self.nsamples, axis=0)				# data through FFT
		self.temp_data_matched_filter[:] = self.temp_data_matched_filter*self.full_matched_filter 	# data through matched filter
		self.temp_data_matched_filter[:] = sp.fftpack.ifft(self.temp_data_matched_filter, axis=0)	# data through IFFT
		max_range_idxs = np.argmax(np.abs(self.temp_data_matched_filter), axis=0)
		max_ranges = (max_range_idxs+self.matched_filter_offset)/self.freq_sampling*self.sound_speed
		if (self.slice_data_at_MF):
			mean_idx = int(round(np.mean(max_range_idxs)))
			lower_idx = mean_idx - self.slice_data_at_MF_size
			if (lower_idx < 0):
				lower_idx = 0
			upper_idx = lower_idx + 2*self.slice_data_at_MF_size
			if (upper_idx > self.nsamples):
				upper_idx = self.nsamples - 1
				lower_idx = upper_idx - 2*self.slice_data_at_MF_size
			if (lower_idx < 0):
				lower_idx = 0
			data = data[lower_idx:upper_idx, :]
		self.cbf_output_max_range = np.mean(max_ranges)
		self.cbf_output_var_range = np.var(max_ranges)
		self.cbf_output_max_value, max_row, max_col = self.beamform(self.cbf_output_heatmap, self.look_azimuths, self.look_elevations, data, self.cbf_filter, self.matched_filter, self.czt_filter, self.nfft, self.num_pos, self.temp_data_cbf_filter)
		self.cbf_output_max_elevation = self.look_elevations[max_row]
		self.cbf_output_max_azimuth = self.look_azimuths[max_col]

	def get_cbf_filter(self, phones, sound_speed, look_azimuths, look_elevations, nfft, f1, f2):
		cbf_filter = np.zeros((len(look_elevations),len(look_azimuths),nfft,phones.shape[0]), order='C', dtype=np.complex128)
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
		Fsignal_flip = np.conj(c(replica))
		matched_filter = Fsignal_flip
		return matched_filter[0,:]

	def get_full_matched_filter(self, replica, num_phones, nfft):
		Freplica = np.conj(sp.fftpack.fft(replica, nfft))
		full_matched_filter = np.tile(np.transpose(Freplica), (1,num_phones))
		return full_matched_filter

	def next_greater_power_of_2(self, x):
		return 2**(x-1).bit_length()

	def get_czt_filter(self, sampling_rate, nfft, f1, f2, filter_size):
		if (self.slice_data_at_MF):
			filter_size = self.slice_data_at_MF_size*2
		w = np.exp(-1j*2*np.pi*(f2-f1)/(nfft*sampling_rate))
		a = np.exp(1j*2*np.pi*f1/sampling_rate)
		czt_filter = CZT(filter_size,nfft,w,a)
		return czt_filter

	def beamform(self, heatmap, look_azimuths, look_elevations, phone_data, cbf_filter, matched_filter, czt_filter, nfft, num_phones, czt_storage_matrix):
		czt_storage_matrix[:] = np.transpose(czt_filter(np.transpose(phone_data)))
		max_val, max_row, max_col = cbf_mac(len(look_elevations), len(look_azimuths), nfft, num_phones, czt_storage_matrix, cbf_filter, matched_filter, heatmap)
		return max_val, max_row, max_col
