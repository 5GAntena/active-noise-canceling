import pyaudio
import numpy as np

from scipy.signal import fftconvolve
from scipy.io import wavfile

from types_lib import _stft, _istft, _amp_to_db, _db_to_amp
from filters import apply_filter

class AudioProcessor:
    def __init__(self, format=pyaudio.paFloat32, channels=2, rate=44100, buffer=2048):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.buffer = buffer
        
        self.p = pyaudio.PyAudio()

        self.input_device = self.p.get_default_input_device_info()
        
        self.input_device_index = self.input_device["index"]
        self.output_device_index = self.find_output_device(self.p)

        self.noise_profile = self.load_noise_file()

        self.input_stream = None
        self.output_stream = None
        
        self.audio_buffer = np.array([], dtype=np.float32)

    def load_noise_file(self):
        sample_rate, noise_data = wavfile.read(r"C:\Users\kemerios\Desktop\tarkov_sounds\factory\amb_factory2-sharedassets3.assets-14.wav")

        noise_data_flatten = noise_data.flatten().astype(np.float32)

        noise_data_picked = noise_data_flatten[: self.buffer * 2]

        return noise_data_picked


    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_data (1d np array): The first parameter.
        noise_data (1d np array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    def noise_sub(self, audio_data, noise_data, n_grad_freq=4, n_grad_time=8, n_fft=2048, win_length=2048, hop_length=1024, n_std_thresh=1.5, prop_decrease=1.0):

        noise_stft = _stft(noise_data, n_fft, hop_length, win_length)
        noise_stft_db = _amp_to_db(np.abs(noise_stft))
        
        mean_freq_noise = np.mean(noise_stft_db, axis=1)
        std_freq_noise = np.std(noise_stft_db, axis=1)
        noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
        
        sig_stft = _stft(audio_data, n_fft, hop_length, win_length)
        sig_stft_db = _amp_to_db(np.abs(sig_stft))
        
        mask_gain_db = np.min(_amp_to_db(np.abs(sig_stft)))
        
        smoothing_filter = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_freq + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_time + 2),
                ]
            )[1:-1],
        )
        
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
        
        db_thresh = np.repeat(
            np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
            np.shape(sig_stft_db)[1],
            axis=0,
        ).T
        
        sig_mask = sig_stft_db < db_thresh
        
        sig_mask = fftconvolve(sig_mask, smoothing_filter, mode="same")
        sig_mask = sig_mask * prop_decrease
        
        sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_db)) * mask_gain_db * sig_mask
        )
        
        sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
        sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)
        
        recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
        
        return recovered_signal

    """Find the output device given the name of the device
    
    Args:
        p (class): PyAudio() object
    
    
    Returns:
        int: Index of the device

    """

    def find_output_device(self, p):

        output_device_index = None

        for i in range(p.get_device_count()):
            if "Voicemeeter Input" in p.get_device_info_by_index(i)["name"]:
                output_device_index = p.get_device_info_by_index(i)["index"]

                break

        return output_device_index

    def start_stream(self):
        self.input_stream = self.p.open(rate=self.rate, 
                                  channels=self.channels, 
                                  format=self.format, 
                                  input=True, 
                                  output=False, 
                                  input_device_index=self.input_device_index, 
                                  frames_per_buffer=self.buffer,
                                  stream_callback=self.audio_callback)
        
        self.output_stream = self.p.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        frames_per_buffer=self.buffer,
                                        output_device_index=self.output_device_index,
                                        input=False,
                                        output=True)

        self.input_stream.start_stream()
        self.output_stream.start_stream()

    def stop_streams(self):
        if self.input_stream is not None:
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if self.output_stream is not None:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        self.p.terminate()

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        denoised_data = self.process_audio(audio_data)

        filtered_data = apply_filter(denoised_data, lowcut_freq=500, filter_type="")
        
        self.output_stream.write(filtered_data.astype(np.float32).tobytes())
        
        return (None, pyaudio.paContinue)
    
    def process_audio(self, audio_data):

        processed_audio = self.noise_sub(audio_data, self.noise_profile)

        return processed_audio

    def run(self):
        try:
            self.start_stream()
            print("Listening...")
            while self.input_stream.is_active():
                pass
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop_streams()