import numpy as np
import librosa
import glob
import os, time

from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile
from multiprocessing import Pool

DTYPE = "float16"
FFT_SIZE = 1024

def stft_wav(path):
    print(path)
    data, sample_rate = librosa.load(path, sr=int(4410))
    spectrogram = librosa.stft(data, dtype=DTYPE, n_fft=FFT_SIZE)
    return spectrogram

def extract_data(l_f, threads):
    pool = Pool(threads)

    return pool.map(stft_wav, l_f)

if __name__ == "__main__":
    files = glob.glob(os.path.join("/data/vision/billf/object-properties/yilundu/hackmit/2017_audio_transfer/Wav_Processing", '*.wav'))

    list_spec = extract_data(files, 32)

    dictionary_list = {str(i): val for (i, val) in enumerate(list_spec)}
    np.savez("training_data", **dictionary_list)
