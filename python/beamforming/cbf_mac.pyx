cimport numpy as np
cimport libc.math as cymath
cimport cython
# cdef extern from "math.h":
# 	float INFINITY

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cbf_mac(unsigned int look_elevations, unsigned int look_azimuths, unsigned int nfft, unsigned int num_phones, np.ndarray[np.complex128_t, ndim=2] Fphone_data, np.ndarray[np.complex128_t, ndim=4] cbf_filter, np.ndarray[np.complex128_t, ndim=1] matched_filter, np.ndarray[np.float_t, ndim=2] output):
	cdef float sum_freqs
	cdef double complex sum_phones
	cdef unsigned int i,j,k,l
	cdef double sum_phones_real, sum_phones_imag

	cdef float max_val = 0.0
	cdef unsigned int max_row = 0
	cdef unsigned int max_col = 0
	for i in range(look_elevations):
		for j in range(look_azimuths):
			sum_freqs = 0.0
			for k in range(nfft):
				sum_phones = 0.0
				for l in range(num_phones):
					sum_phones = sum_phones + Fphone_data[k,l]*cbf_filter[i,j,k,l]*matched_filter[k]
				sum_phones_real = sum_phones.real
				sum_phones_imag = sum_phones.imag
				sum_freqs = sum_freqs + cymath.sqrt(cymath.pow(sum_phones_real,2) + cymath.pow(sum_phones_imag,2))
			output[i,j] = sum_freqs
			if sum_freqs > max_val:
				max_val = sum_freqs
				max_row = i
				max_col = j
	return max_val, max_row, max_col