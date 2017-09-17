import matplotlib.pyplot as plt
import numpy as np
import librosa
import glob
import os, time
from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile

DTYPE = "float16"
FFT_SIZE = 512
MAGNITUDE_SCALING = 30


def split_and_fft(path):
    start_time = time.time()
    files = glob.glob(os.path.join(path, '*.wav'))
    #files = ["test.wav
    list_spec = []

    num = 0
    for file in files:
        data, sample_rate = librosa.load('test0.wav', sr=int(8000))
        #freq = data.T[0]
 
        spectrogram = librosa.stft(data, dtype=DTYPE, n_fft=FFT_SIZE)

        #print(spectrogram.shape)

        #np.savez_compressed(str(num), spectrogram, delimiter=',')

        print(str(num) + " : " + str(time.time() - start_time))

        num+=1

    print(spectrogram.shape)
    list_spec.append(spectrogram)

    np.savez("training_data", list_spec, delimiter=',')

    return spectrogram, sample_rate

def invert_wav(spectrogram, sample_rate, iterations):
    mag, actual_phase = librosa.magphase(spectrogram)

    # Try to reconstruct the phase using the iterative algorithm above.
    phase = np.exp(1.j * np.random.uniform(0., 2*np.pi, size=actual_phase.shape))
    x_ = librosa.istft((mag*MAGNITUDE_SCALING) * phase)

    for i in range(iterations+1):
        _, phase = librosa.magphase(librosa.stft(x_, n_fft=FFT_SIZE))
        x_ = librosa.istft((mag*MAGNITUDE_SCALING) * phase)

    #print(sample_rate)
    wavfile.write("invert2.wav", int(sample_rate), x_)

specs, sample_rate = split_and_fft(path=os.path.dirname(os.path.abspath(__file__)))
invert_wav(specs, sample_rate, iterations=10)