import numpy as np
from torch.autograd import Variable
import torch

from model import RecurrentEncoder, RecurrentDecoder, sample_z
import hyperparameters as hp

input_dim = hp.input_dim
latent_dim = hp.latent_dim
lstm_layers = hp.lstm_layers

def transform_song(input_dat, enc_path, dec_path):
    enc = RecurrentEncoder(input_dim, latent_dim, lstm_layers)
    dec = RecurrentDecoder(latent_dim, input_dim, lstm_layers)

    enc.load_state_dict(torch.load(enc_path))
    dec.load_state_dict(torch.load(dec_path))

    enc.eval()
    dec.eval()
    enc.init_hidden(input_dim, on_gpu=False)
    dec.init_hidden(latent_dim, on_gpu=False)

    input_dat = input_dat.T
    length, freq = input_dat.shape
    out_dat = np.zeros((length, freq))

    input_var = torch.from_numpy(input_dat.astype(np.float))
    input_var = Variable(input_var).float()

    z_mu, z_var = enc(input_var)
    z = sample_z(z_mu, z_var, 1, latent_dim, on_gpu=False)
    X_sample = dec(z)

    out_dat = X_sample.data.numpy()

    return out_dat.T


if __name__ == "__main__":
    enc_path = "classical_enc.mdl"
    dec_path = "classical_dec.mdl"
    input_dat = np.load("training_data.npz")['0']
    output_dat = transform_song(input_dat, enc_path, dec_path)
    np.save("output_vae", output_dat)

