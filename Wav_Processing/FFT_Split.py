import matplotlib.pyplot as plt
import numpy as np
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
        sample_rate, data = wavfile.read(file) # load the data
        freq = data.T[0]

        f, t, Zxx = signal.stft(np.array(freq), window="")
        np.savez_compressed(str(num), Zxx, delimiter=',')

        num+=1
    return Zxx, sample_rate

def invert_wav(Zxx, sample_rate):
    t, x = signal.istft(Zxx)
    wavfile.write("invert.wav", sample_rate, x)

Zxx, sample_rate = split_and_fft(path=os.path.dirname(os.path.abspath(__file__)),sample=1)
invert_wav(Zxx, sample_rate)

