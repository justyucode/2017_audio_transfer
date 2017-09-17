import matplotlib.pyplot as plt
import numpy as np
import librosa
import glob
import os, time
from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile

DTYPE = "float16"
FFT_SIZE = 1024
GAIN = 5
ITERATIONS = 10


def split_and_fft(path):
    data, sample_rate = librosa.load(file, sr=int(4410))
    spectrogram = librosa.stft(data, dtype=DTYPE, n_fft=FFT_SIZE)
    list_spec.append(spectrogram)

    return spectrogram, sample_rate

def invert_wav(spectrogram, sample_rate, iterations):
    mag, actual_phase = librosa.magphase(spectrogram)

    # Try to reconstruct the phase using the iterative algorithm above.
    phase = np.exp(1.j * np.random.uniform(0., 2*np.pi, size=actual_phase.shape))
    x_ = librosa.istft((GAIN * mag) * phase)

    for i in range(iterations+1):
        _, phase = librosa.magphase(librosa.stft(x_, n_fft=FFT_SIZE))
        x_ = librosa.istft((GAIN * mag) * phase)

    wavfile.write("sounds/recreated.wav", int(sample_rate), x_)