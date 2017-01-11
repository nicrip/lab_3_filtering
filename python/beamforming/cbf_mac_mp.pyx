cimport numpy as np
cimport libc.math as cymath
cimport cython
from cython.parallel import prange
# cdef extern from "math.h":
# 	float INFINITY

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cbf_mac(unsigned int look_elevations, unsigned int look_azimuths, unsigned int nfft, unsigned int num_phones, np.ndarray[np.complex128_t, ndim=2] Fphone_data, np.ndarray[np.complex128_t, ndim=4] cbf_filter, np.ndarray[np.complex128_t, ndim=1] matched_filter, np.ndarray[np.float_t, ndim=2] output):
	cdef double sum_freqs
	cdef double complex sum_phones
	cdef int i,j,k,l
	cdef double sum_phones_real, sum_phones_imag

	cdef np.float_t max_val = 0.0
	cdef int max_row = 0
	cdef int max_col = 0

	for i in prange(look_elevations, nogil=True, schedule='static'):
		for j in range(look_azimuths):
			sum_phones_real = 0.0
			sum_phones_imag = 0.0
			sum_freqs = 0.0
			for k in range(nfft):
				sum_phones = 0.0
				for l in range(num_phones):
					sum_phones = sum_phones + Fphone_data[k,l]*cbf_filter[i,j,k,l]*matched_filter[k]
				sum_phones_real = sum_phones.real
				sum_phones_imag = sum_phones.imag
				sum_freqs = sum_freqs + cymath.sqrt(cymath.pow(sum_phones_real,2) + cymath.pow(sum_phones_imag,2))
			output[i,j] = <np.float_t>sum_freqs

	for i in range(look_elevations):
		for j in range(look_azimuths):
			if output[i,j] > max_val:
				max_val = output[i,j]
				max_row = i
				max_col = j
	return max_val, max_row, max_col
