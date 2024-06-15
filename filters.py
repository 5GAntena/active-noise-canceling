
import numpy as np


def fft_highpass_filter(in_data, cutoff_freq, rate):

    fft_signal = np.fft.fft(in_data)
    frequency = np.fft.fftfreq(len(in_data), d=1.0 / rate)

    high_pass_filter = np.abs(frequency) > cutoff_freq
    filtered_fft_signal = fft_signal * high_pass_filter

    filtered_signal = np.fft.ifft(filtered_fft_signal)

    return np.real(filtered_signal)


def fft_lowpass_filter(in_data, cutoff_freq, rate):

    fft_signal = np.fft.fft(in_data)
    frequency = np.fft.fftfreq(len(in_data), d=1.0 / rate)

    low_pass_filter = np.abs(frequency) <= cutoff_freq
    filtered_fft_signal = fft_signal * low_pass_filter

    filtered_signal = np.fft.ifft(filtered_fft_signal)
    
    return np.real(filtered_signal)

def fft_bandpass_filter(in_data, lowcut_freq, highcut_freq, rate):

    fft_signal = np.fft.fft(in_data)
    frequency = np.fft.fftfreq(len(in_data), d=1.0 / rate)

    bandpass_filter = (np.abs(frequency) >= lowcut_freq) & (np.abs(frequency) <= highcut_freq)
    filtered_fft_signal = fft_signal * bandpass_filter

    filtered_signal = np.fft.ifft(filtered_fft_signal)
    
    return np.real(filtered_signal)