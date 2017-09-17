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


def split_and_fft(path):
    start_time = time.time()
    # files = glob.glob(os.path.join(path, '*.wav'))
    files = ["test0.wav"]
    list_spec = []

    num = 0

    for file in files:
        data, sample_rate = librosa.load(file, sr=int(4410))
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

def invert_wav(spectrogram, sample_rate, iterations, name):
    mag, actual_phase = librosa.magphase(spectrogram)

    # Try to reconstruct the phase using the iterative algorithm above.
    phase = np.exp(1.j * np.random.uniform(0., 2*np.pi, size=actual_phase.shape))
    x_ = librosa.istft((GAIN * mag) * phase)

    for i in range(iterations+1):
        _, phase = librosa.magphase(librosa.stft(x_, n_fft=FFT_SIZE))
        x_ = librosa.istft((GAIN * mag) * phase)

    #print(sample_rate)
    wavfile.write(name + ".wav", int(sample_rate), x_)

# specs, sample_rate = split_and_fft(path="./")
sample_rate = 4410 / 10

specs_input= np.load("input_vae.npy")
specs_output = np.load("output_vae.npy")

specs = specs_output * 100
invert_wav(specs, sample_rate, 10, "output")
invert_wav(specs_input, sample_rate * 10, 10, "input")
