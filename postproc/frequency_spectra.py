# -*- coding: utf-8 -*-
"""
@author: J. Massey
@description: Module to determine the convergence of frequency spectra
@contact: jmom1n15@soton.ac.uk
"""

# Imports
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from collections import deque

# Functions
def freq_spectra(t, u, **kwargs):
	"""
	Returns the FFT of u together with the associated frequency after resampling the signal evenly.
	:param t: Time series.
	:param u: Signal series.
	:param kwargs:
		resample: Boolean to resample the signal evenly spaced in time.
		lowpass: Boolean to apply a low-pass filter to the transformed signal.
		windowing: Boolean to apply a windowing function to the temporal signal.
		downsample: Integer (where 0=False) for the number of points to average on the downsampling procedure.
		expanding_windowing: Boolean that changes the output to a list of uk and freqs representing an expanding
							window so that we can see convergence to a manifold
	:return: freqs 1D array and uk 1D array.
	"""
	resample = kwargs.get('resample', True)
	lowpass = kwargs.get('lowpass', False)
	windowing = kwargs.get('windowing', True)
	tukey_windowing = kwargs.get('tukey_windowing', False)
	downsample = kwargs.get('downsample', False)

	if tukey_windowing:
		windowing = False
		u = _tuk_window(u)
	# Re-sample u on a evenly spaced time series (constant dt)
	if resample:
		u = u - np.mean(u)
		u_function = interp1d(t, u, kind='cubic')
		t_min, t_max = np.min(t), np.max(t)
		dt = (t_max - t_min) / len(t)
		t_regular = np.arange(t_min, t_max, dt)[:-1]  # Skip last one because can be problematic if > than actual t_max
		u = u_function(t_regular)
	else:
		dt = t[1] - t[0]
		t_min, t_max = np.min(t), np.max(t)

	if windowing: u = _window(u)  # Windowing
	if lowpass: u = _low_pass_filter(u)  # Signal filtering for high frequencies

	# Compute power fft and associated frequencies
	# uk = np.fft.fft(u)/u.size
	uk = (1 / (t_max - t_min)) * np.fft.fft(u)
	# uk = (dt/u.size)*np.fft.fft(u)
	uk = uk*np.conj(uk)
	uk = uk.real
	freqs = np.fft.fftfreq(uk.size, d=dt)

	# Downsample averaging
	if downsample > 0:
		uk = _downsample_avg(uk, downsample)
		freqs = _downsample_avg(freqs, downsample)
		uk = uk[:-1]
		freqs = freqs[:-1]

	# Take only positive frequencies and return arrays
	freqs = freqs[freqs > 0]
	uk = uk[:len(freqs)]
	return freqs, uk


def freq_spectra_convergence(t, u, n=5, OL=0.5, **kwargs):
	"""
	This is a function that takes a hann window of signal at descrete intervals and returns the frequency spec
	for each window, as well as the mean and var for these ffts
	Args:
		t: Time series
		u: Signal Series
		**kwargs:
			resample: Boolean to resample the signal evenly spaced in time.
			lowpass: Boolean to apply a low-pass filter to the transformed signal.
			windowing: Boolean to apply a windowing function to the temporal signal.
			downsample: Integer (where 0=False) for the number of points to average on the downsampling procedure.
	:return: uks: list of

	Returns:
			uks: n dimensional list of (label, uk) tuple where the label is the window
			fs : n dimensional list of frequencies
			means: n dimensional list of windowed bin mean
			variance: n dimensional list of windowed bin variance
	"""
	import numpy as np
	from scipy.interpolate import interp1d
	from collections import deque

	u_partial_ol_list = _split_overlap(u, n, OL)
	t_partial_ol_list = _split_overlap(t, n, OL)

	uk_partial_ol_list = deque(); freqs_partial_ol_list = deque()
	means = deque(); variances = deque()
	for idx, tup in enumerate(list(zip(t_partial_ol_list, u_partial_ol_list))):
		freqs, uk = freq_spectra(tup[0], tup[1], resample=False, **kwargs)
		means.append(np.mean(uk)); variances.append(np.var(uk))
		freqs_partial_ol_list.append(freqs); uk_partial_ol_list.append((f"Bin {idx+1}", uk))
	return list(uk_partial_ol_list), list(freqs_partial_ol_list), list(means), list(variances)

def freq_spectra_ensembling(t, u, n, OL=0.5, **kwargs):
	"""
	This is a function that takes a hann window of signal at descrete intervals and returns the frequency spec
	for each window, as well as the mean and var for these ffts. It then takes an ensembled average of these
	windows to show the convergence of a spectra.
	Args:
		t: Time series
		u: Signal Series
		**kwargs:
			resample: Boolean to resample the signal evenly spaced in time.
			lowpass: Boolean to apply a low-pass filter to the transformed signal.
			windowing: Boolean to apply a windowing function to the temporal signal.
			downsample: Integer (where 0=False) for the number of points to average on the downsampling procedure.
	:return: uks: list of

	Returns:
			uks: n dimensional list of (label, uk) tuple where the label is the window
			fs : n dimensional list of frequencies
	"""

	u = u - np.mean(u)
	u_function = interp1d(t, u, kind='cubic')
	t_min, t_max = np.min(t), np.max(t)
	dt = (t_max - t_min) / len(t)
	t = np.arange(t_min, t_max, dt)[:-1]  # Regularize t and Skip last because can be problematic if > than actual t_max
	u = u_function(t)  # Regularize u

	u_partial_ol_list = _split_overlap(u, n, OL)
	t_partial_ol_list = _split_overlap(t, n, OL)

	uk_ensemble_ol_list = deque(); freqs_ensemble_ol_list = deque()
	tmp_uk = deque(); tmp_f = deque()
	for idx, tup in enumerate(list(zip(t_partial_ol_list, u_partial_ol_list))):
		freqs, uk = freq_spectra(tup[0], tup[1], resample=False, **kwargs)
		tmp_uk.append(uk); tmp_f.append(freqs)
		uk_ensemble_ol_list.append((f"Window ${idx+1}$", np.average(np.array(tmp_uk), axis=0)))
		freqs_ensemble_ol_list.append(np.average(np.array(tmp_f), axis=0))
	return list(uk_ensemble_ol_list), list(freqs_ensemble_ol_list)


class FreqConv:
	"""
	This is a class that holds functions to determine the convergence of a time series using the ensembled freq
	spectra.
	"""
	def __init__(self, t, u, n=6, OL=0.5):
		self.t = t
		self.u = u
		self.n = n
		self.OL = OL

	def freq(self, t, u, **kwargs):
		"""
		Returns the FFT of u together with the associated frequency after resampling the signal evenly.
		:param t: Time series.
		:param u: Signal series.
		:param kwargs:
			resample: Boolean to resample the signal evenly spaced in time.
			lowpass: Boolean to apply a low-pass filter to the transformed signal.
			windowing: Boolean to apply a windowing function to the temporal signal.
			downsample: Integer (where 0=False) for the number of points to average on the downsampling procedure.
			expanding_windowing: Boolean that changes the output to a list of uk and freqs representing an expanding
								window so that we can see convergence to a manifold
		:return: freqs 1D array and uk 1D array.
		"""
		resample = kwargs.get('resample', True)
		lowpass = kwargs.get('lowpass', False)
		windowing = kwargs.get('windowing', True)
		tukey_windowing = kwargs.get('tukey_windowing', False)
		downsample = kwargs.get('downsample', False)

		if tukey_windowing:
			windowing = False
			u = _tuk_window(self.u)
		# Re-sample u on a evenly spaced time series (constant dt)
		if resample:
			u, t = self._resample(t, u)
			t_min, t_max = np.min(t), np.max(t)
			dt = (t_max - t_min) / len(t)
		else:
			t_min, t_max = np.min(t), np.max(t)
			dt = (t_max - t_min) / len(t)

		if windowing: u = self._window(u)  # Windowing
		if lowpass: u = _low_pass_filter(u)  # Signal filtering for high frequencies

		# Compute power fft and associated frequencies
		uk = 1/(t_max-t_min) * np.fft.fft(u)
		freqs = np.fft.fftfreq(uk.size, d=dt)
		uk = uk * np.conj(uk)
		uk = uk.real

		# Downsample averaging
		if downsample > 0:
			uk = _downsample_avg(uk, downsample)
			freqs = _downsample_avg(freqs, downsample)
			uk = uk[:-1]
			freqs = freqs[:-1]

		# Take only nyquist frequency and get rid of sampling rate
		freqs = freqs[freqs > 0]
		uk = uk[:len(freqs)]
		return freqs, abs(uk)

	def ensemble(self, **kwargs):
		"""
		This is a function that takes a Hann window of signal at discrete intervals and returns the frequency spec
		for each window, as well as the mean and var for these ffts. It then takes an ensembled average of these
		windows to show the convergence of a spectra.
		Args:
			t: Time series
			u: Signal Series
			**kwargs:
				resample: Boolean to resample the signal evenly spaced in time.
				lowpass: Boolean to apply a low-pass filter to the transformed signal.
				windowing: Boolean to apply a windowing function to the temporal signal.
				downsample: Integer (where 0=False) for the number of points to average on the downsampling procedure.
		:return: uks: list of

		Returns:
				uks: n dimensional list of (label, uk) tuple where the label is the window
				fs : n dimensional list of frequencies
		"""
		t, u = self._resample(self.t, self.u)
		u_partial_ol_list = self._split_overlap(u)
		t_partial_ol_list = self._split_overlap(t)

		uk_ensemble_avg = deque(); freqs_ensemble_avg = deque()
		area = deque(); tmp_f = deque(); tmp_uk = deque(); ind_wind_area = deque()
		for idx, tup in enumerate(list(zip(t_partial_ol_list, u_partial_ol_list))):
			# Can't resample all at once because you need the whole signal
			t, u = tup[0], tup[1]
			freqs, uk = self.freq(t=t, u=u, resample=False, windowing=True, **kwargs)
			ind_wind_area.append(np.trapz(uk, freqs))
			tmp_f.append(freqs); tmp_uk.append(uk)
			uk_mean = np.mean(np.array(tmp_uk), axis=0)
			uk_ensemble_avg.append((f"Window ${idx+1}$", uk_mean))
			freqs_mean = np.mean(np.array(tmp_f), axis=0)
			freqs_ensemble_avg.append(freqs_mean)
			area.append(np.trapz(uk_mean, freqs_mean))
		return np.array(uk_ensemble_avg), np.array(freqs_ensemble_avg), np.array(area)

	def f_conv(self, cycles, **kwargs):
		"""
		Find how the spectra converges to a steady value
		Args:
			cycles: The number of convection cycles to split the TS over
			**kwargs: arguments for parent methods

		Returns:
			RMS: RMS difference between adjacent spectra normalised by the integral of that spectra
			t: 1d numpy array holding the times the windows are centred around

		"""
		labelled_uks_e, fs_e, area = self.ensemble(**kwargs)
		uks_e = deque()
		for loop1 in labelled_uks_e:
			uks_e.append(loop1[1])

		uks_e = np.array(uks_e)

		off1 = uks_e[1:]
		off2 = uks_e[:-1]

		diff_rms = np.sqrt((off1 - off2) ** 2)
		rms = deque()
		for loop in zip(diff_rms, fs_e):
			rms.append(np.trapz(loop[0], loop[1]))
		rms = np.array(rms)

		normed_error = (rms / area[1:])

		window_t = np.linspace(min(self.t) + cycles, max(self.t), len(normed_error))
		return normed_error, window_t

	def welch(self, **kwargs):
		"""
		Returns the FFT of u together with the associated frequency after resampling the signal evenly.
		In this case, an averages of the spectras is computed.
		:param t: Time series.
		:param u:  Signal series.
		:param n:  Number of splits of the original whole time signal.
		:param OL: Overlap of the splits to compute the time
		:param kwargs:
			lowpass: Boolean to apply a low-pass filte to the transformed signal.
			windowing: Boolean to apply a windowing function to the temporal signal.
			expanding_windowing: Boolean that changes the output to a list of uk and freqs representing an expanding window
			evaluation of the frequency signals
		:return: freqs 1D array and uk 1D array.
		"""
		# Re-sample u on a evenly spaced time series (constant dt)
		t, u = self._resample(self.t, self.u)

		u_partial_ol_list = _split_overlap(u, self.n, self.OL)
		t_partial_ol_list = _split_overlap(t, self.n, self.OL)

		uk_partial_ol_list = deque()
		freqs_partial_ol_list = deque()
		for tup in list(zip(t_partial_ol_list, u_partial_ol_list)):
			freqs, uk = self.freq(t=tup[0], u=tup[1], resample=False, windowing=True, **kwargs)
			freqs_partial_ol_list.append(freqs)
			uk_partial_ol_list.append(uk)
		uk_mean = np.mean(np.array(uk_partial_ol_list), axis=0)
		freqs_mean = np.mean(np.array(freqs_partial_ol_list), axis=0)
		return freqs_mean, uk_mean

	def _resample(self, t, u):
		"""
		Regularise the time interval and fit cubic function to u.
		Returns:
			Resampled u,t
		"""
		u = u - np.mean(u)
		u_function = interp1d(t, u, kind='cubic')
		t_min, t_max = np.min(t), np.max(t)
		dt = (t_max - t_min) / len(t)
		t = np.arange(t_min, t_max, dt)[:-1]  # Regularize t and Skip last, can be problematic if > than t_max
		u = u_function(t)  # Regularize u
		return t, u

	def _split_overlap(self, a):
		"""
		:param a: array to split and overlap.
		:param n: number of splits of a.
		:param OL: overlap.
		:return: c, a list of the splits of a in function of n and OL
		"""
		splits_size = int(round(a.size / self.n))
		nOL = int(round(splits_size * self.OL))
		skip = splits_size - nOL
		b = [a[i: i + splits_size] for i in range(0, len(a), skip)]
		c = deque()
		for i, item in enumerate(b):
			if len(item) == splits_size:
				c.append(item)
		return np.array(c)

	def _window(self, a):
		w = signal.windows.hann(len(a))
		return a * w

	def _tuk_window(self, a, alpha=0.5):
		w = signal.windows.tukey(len(a), alpha=alpha)
		return a * w

	def _downsample_avg(self, arr, n):
		"""
		Average every n elements a 1D array.
		:param arr: 1D array.
		:param n: size of the averaging subarray.
		:return: Downsampled-averaged 1D array.
		"""
		end = n * int(len(arr) / n)
		return np.mean(arr[:end].reshape(-1, n), 1)

	def _downsample_simple(self, arr, n):
		"""
		Skip n elements of a 1D array.
		:param arr: 1D array.
		:param n: integer which defines the skips.
		:return: Downsampled 1D array.
		"""
		return arr[::n]

	def _low_pass_filter(self, u):
		"""
		Apply a low-pass filter to u.
		:param u: Temporal signal 1D.
		:return: Windowed signal.
		"""
		b, a = signal.butter(3, 0.4, 'low')  # 2nd arg: Fraction of fs that wants to be filtered
		return signal.filtfilt(b, a, u)


def freq_spectra_Welch(t, u, n=4, OL=0.5, **kwargs):
	"""
	Returns the FFT of u together with the associated frequency after resampling the signal evenly.
	In this case, an averages of the spectras is computed.
	:param t: Time series.
	:param u:  Signal series.
	:param n:  Number of splits of the original whole time signal.
	:param OL: Overlap of the splits to compute the time
	:param kwargs:
		lowpass: Boolean to apply a low-pass filte to the transformed signal.
		windowing: Boolean to apply a windowing function to the temporal signal.
		expanding_windowing: Boolean that changes the output to a list of uk and freqs representing an expanding window
		evaluation of the frequency signals
	:return: freqs 1D array and uk 1D array.
	"""
	import numpy as np
	from scipy.interpolate import interp1d

	# Re-sample u on a evenly spaced time series (constant dt)
	u = u - np.mean(u)
	u_function = interp1d(t, u, kind='cubic')
	t_min, t_max = np.min(t), np.max(t)
	dt = (t_max - t_min) / len(t)
	t = np.arange(t_min, t_max, dt)[:-1]  # Regularize t and Skip last because can be problematic if > than actual t_max
	u = u_function(t)  # Regularize u

	u_partial_ol_list = _split_overlap(u, n, OL)
	t_partial_ol_list = _split_overlap(t, n, OL)

	uk_partial_ol_list = []
	freqs_partial_ol_list = []
	for tup in list(zip(t_partial_ol_list, u_partial_ol_list)):
		freqs, uk = freq_spectra(tup[0], tup[1], resample=False, **kwargs)
		freqs_partial_ol_list.append(freqs)
		uk_partial_ol_list.append(uk)
	uk_mean = np.mean(uk_partial_ol_list, axis=0)
	freqs_mean = np.mean(freqs_partial_ol_list, axis=0)
	return freqs_mean, uk_mean


def freq_spectra_scipy_welch(t, u, n, OL, **kwargs):
	import numpy as np
	from scipy.interpolate import interp1d
	# Re-sample u on a evenly spaced time series (constant dt)
	u = u - np.mean(u)
	u_function = interp1d(t, u, kind='cubic')
	t_min, t_max = np.min(t), np.max(t)
	dt = (t_max - t_min) / len(t)
	t = np.arange(t_min, t_max, dt)[
		:-1]  # Regularize t and Skip last one because can be problematic if > than actual t_max
	u = u_function(t)  # Regularize u

	# Buggy (do not use)
	# freqs, uk = signal.welch(u, fs=1 / dt, window='hanning', nperseg=int(u.size / n), noverlap=None, scaling='spectrum')

	# Bug fix
	nperseg = int(u.size / n)
	noverlap = int(nperseg * OL)
	freqs, uk = signal.welch(u, fs=1 / dt, window='hanning', nperseg=nperseg, noverlap=noverlap, scaling='spectrum')

	return freqs, uk


def _split_overlap(a, n, OL):
	"""
	:param a: array to split and overlap.
	:param n: number of splits of a.
	:param OL: overlap.
	:return: c, a list of the splits of a in function of n and OL
	"""
	# Problem here with the window size allocation
	splits_size = int(round(a.size / n))
	nOL = int(round(splits_size * OL))
	skip = splits_size - nOL
	b = [a[i: i + splits_size] for i in range(0, len(a), skip)]
	c = []
	for i, item in enumerate(b):
		if len(item) == splits_size:
			c.append(item)
	return c


def _window(a):
	w = signal.windows.hann(len(a))
	return a * w


def _tuk_window(a, alpha=0.5):
	w = signal.windows.tukey(len(a), alpha=alpha)
	return a * w


def _downsample_avg(arr, n):
	"""
	Average every n elements a 1D array.
	:param arr: 1D array.
	:param n: size of the averaging subarray.
	:return: Downsampled-averaged 1D array.
	"""
	end = n * int(len(arr) / n)
	return np.mean(arr[:end].reshape(-1, n), 1)


def _downsample_simple(arr, n):
	"""
	Skip n elements of a 1D array.
	:param arr: 1D array.
	:param n: integer which defines the skips.
	:return: Downsampled 1D array.
	"""
	return arr[::n]


def _low_pass_filter(u):
	"""
	Apply a low-pass filter to u.
	:param u: Temporal signal 1D.
	:return: Windowed signal.
	"""
	b, a = signal.butter(3, 0.4, 'low')  # 2nd arg: Fraction of fs that wants to be filtered
	return signal.filtfilt(b, a, u)
