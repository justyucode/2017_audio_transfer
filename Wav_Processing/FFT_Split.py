import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile

def split_and_fft(path, sample):
    files = glob.glob(os.path.join(path, '*.wav'))

    num = 0
    for file in files:

        fs, data = wavfile.read(file) # load the data

        freq = data.T[0] # this is a two channel soundtrack, I get the first track

        bitrate = 16
        norm_freq =[(ele/2**bitrate)*2-1 for ele in freq]

        f, t, Sxx = signal.spectrogram(np.array(norm_freq))
        np.savez_compressed(str(num), Sxx, delimiter=',')

        num+=1

split_and_fft(path=os.path.dirname(os.path.abspath(__file__)),sample=1000)
