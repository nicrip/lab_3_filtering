import numpy as np
import scipy as sp  # FASTEST IFFT!
cimport numpy as np
cimport libc.math as cymath
cimport cython
# cdef extern from "math.h":
# 	float INFINITY

# example of using pyfftw
# import pyfftw
# a = pyfftw.empty_aligned(nfft, dtype='complex128')
# b = pyfftw.empty_aligned(nfft, dtype='complex128')
# ifft_object = pyfftw.FFTW(a, b, direction='FFTW_BACKWARD')
# a[k] = sum_phones
# output[i,j] = np.max(np.abs(ifft_object()))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cbf_mac(unsigned int look_elevations, unsigned int look_azimuths, unsigned int nfft, unsigned int num_phones, np.ndarray[np.complex128_t, ndim=2] Fphone_data, np.ndarray[np.complex128_t, ndim=4] cbf_filter, np.ndarray[np.complex128_t, ndim=1] matched_filter, np.ndarray[np.float_t, ndim=2] output):
	cdef double max_time, curr_time
	cdef np.complex128_t sum_phones
	cdef unsigned int i,j,k,l
	# cdef np.ndarray[np.complex128_t, ndim=2] ifft_input = np.zeros((look_azimuths,nfft), order='C', dtype=np.complex128)
	cdef double complex[:,:] ifft_input = np.zeros((look_azimuths,nfft), order='C', dtype=np.complex128)
	cdef double complex ifft_val
	cdef double ifft_real, ifft_imag
	
	cdef double max_val = 0.0
	cdef unsigned int max_row = 0
	cdef unsigned int max_col = 0
	for i in range(look_elevations):
		for j in range(look_azimuths):
			for k in range(nfft):
				sum_phones = 0.0
				for l in range(num_phones):
					sum_phones = sum_phones + Fphone_data[k,l]*cbf_filter[i,j,k,l]*matched_filter[k]
				ifft_input[j,k] = sum_phones
			
		sp.fftpack.ifft(ifft_input, overwrite_x=True)
		for j in range(look_azimuths):
			max_time = 0.0
			for k in range(nfft):
				ifft_val = ifft_input[j,k]
				ifft_real = ifft_val.real
				ifft_imag = ifft_val.imag
				curr_time = (ifft_real * ifft_real) + (ifft_imag * ifft_imag)
				# curr_time = cymath.sqrt(cymath.pow(ifft_real,2) + cymath.pow(ifft_imag,2))
				if curr_time > max_time:
					max_time = curr_time
			output[i,j] = max_time
			if max_time > max_val:
				max_val = max_time
				max_row = i
				max_col = j
	return max_val, max_row, max_col