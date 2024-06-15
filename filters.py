
import numpy as np

def apply_filter(in_data, lowcut_freq=0, highcut_freq=0, rate=44100, filter_type=""):

    fft_signal = np.fft.fft(in_data)
    frequency = np.fft.fftfreq(len(in_data), d=1.0 / rate)

    match filter_type:
        case "high":
            filter_pass = np.abs(frequency) > highcut_freq

            filtered_fft_signal = fft_signal * filter_pass
            filtered_signal = np.fft.ifft(filtered_fft_signal)

            return np.real(filtered_signal)

        case "low":
            filter_pass = np.abs(frequency) <= lowcut_freq

            filtered_fft_signal = fft_signal * filter_pass
            filtered_signal = np.fft.ifft(filtered_fft_signal)

            return np.real(filtered_signal)
        
        case "band":
            filter_pass = (np.abs(frequency) >= lowcut_freq) & (np.abs(frequency) <= highcut_freq)

            filtered_fft_signal = fft_signal * filter_pass
            filtered_signal = np.fft.ifft(filtered_fft_signal)

            return np.real(filtered_signal)
        
        case "":
            return in_data