import matplotlib.pyplot as plt
import numpy as np
import librosa
import glob
import os
from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile

def split_and_fft(path, sample):
    # files = glob.glob(os.path.join(path, '*.wav'))
    files = ["test.wav"]

    num = 0
    for file in files:
        data, sample_rate = librosa.load('test.wav', sr=44100)
        #freq = data.T[0]

        f, t, Sxx = signal.spectrogram(data)
        #spectrogram = librosa.stft(data)

        #np.savez_compressed(str(num), spectrogram, delimiter=',')

        num+=1

    return Sxx, sample_rate

def invert_wav(spectrogram, sample_rate, iterations):
    mag, actual_phase = librosa.magphase(spectrogram)

    # Try to reconstruct the phase using the iterative algorithm above.
    phase = np.exp(1.j * np.random.uniform(0., 2*np.pi, size=actual_phase.shape))
    _, x_ = signal.ispectrogram(mag * phase)

    for i in range(iterations+1):
        _,_,specs = signal.spectrogram(x_)
        _, phase = librosa.magphase(specs)
        _, x_ = signal.ispectrogram(mag * phase)

    wavfile.write("invert.wav", sample_rate, x_)

specs, sample_rate = split_and_fft(path=os.path.dirname(os.path.abspath(__file__)),sample=1)
invert_wav(specs, sample_rate, iterations=1)