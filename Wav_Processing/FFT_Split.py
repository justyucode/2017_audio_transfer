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
		
		a = data.T[0] # this is a two channel soundtrack, I get the first track

		bitrate = 16
		b=[(ele/2**bitrate)*2-1 for ele in a] # this is bitrate-bit track, b is now normalized on [-1,1)

		f, t, Sxx = signal.spectrogram(np.array(b))
		np.savez_compressed(str(num), Sxx, delimiter=',')

		num+=1

split_and_fft(path=os.path.dirname(os.path.abspath(__file__)),sample=1000)