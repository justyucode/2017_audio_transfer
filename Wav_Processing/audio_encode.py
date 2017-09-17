import matplotlib.pyplot as plt
import numpy as np
import librosa
import glob
import os, time
from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile

DTYPE = "float16"
FFT_SIZE = 2048

def split_and_fft(path):
    start_time = time.time()
    files = glob.glob(os.path.join(path, '*.wav'))
    #files = ["test.wav
    list_spec = []

    num = 0
    for file in files:
        data, sample_rate = librosa.load('test.wav', sr=int(44100/20))
        #freq = data.T[0]
 
        spectrogram = librosa.stft(data, dtype=DTYPE, n_fft=FFT_SIZE)

        #print(spectrogram.shape)

        #np.savez_compressed(str(num), spectrogram, delimiter=',')

        print(str(num) + " : " + str(time.time() - start_time))

        num+=1

    list_spec.append(spectrogram)

    np.savez_compressed("training_data", list_spec, delimiter=',')

    return spectrogram, sample_rate

def invert_wav(spectrogram, sample_rate, iterations):
    mag, actual_phase = librosa.magphase(spectrogram)

    # Try to reconstruct the phase using the iterative algorithm above.
    phase = np.exp(1.j * np.random.uniform(0., 2*np.pi, size=actual_phase.shape))
    x_ = librosa.istft(mag * phase)

    for i in range(iterations+1):
        _, phase = librosa.magphase(librosa.stft(x_, n_fft=FFT_SIZE))
        x_ = librosa.istft(mag * phase)

    #print(sample_rate)
    wavfile.write("invert.wav", int(sample_rate), x_)

#specs, sample_rate = split_and_fft(path=os.path.dirname(os.path.abspath(__file__)))
specs = np.load("../output_vae.npy")
sample_rate = 513
invert_wav(specs, sample_rate, iterations=10)
