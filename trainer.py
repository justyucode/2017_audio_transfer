import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from random import shuffle
from torch.autograd import Variable

from model import RecurrentEncoder, RecurrentDecoder, sample_z
import hyperparameters as hp



# =============================== TRAINING ====================================
npz = np.load("training_data.npz")

input_dim = hp.input_dim
latent_dim = hp.latent_dim
lstm_layers = hp.lstm_layers
num_epoch = 10
gpu_id = 3
batch_size = 200

enc = RecurrentEncoder(input_dim, latent_dim, lstm_layers).cuda(gpu_id)
dec = RecurrentDecoder(latent_dim, input_dim, lstm_layers).cuda(gpu_id)

solver_enc = optim.Adam(enc.parameters(), lr=1e-4)
solver_dec = optim.Adam(dec.parameters(), lr=1e-4)
print("At least this should probably run")

for it in range(num_epoch):
    files = npz.files
    shuffle(files)

    for file in files:
        dat = npz[file].T
        length, embed = dat.shape
        k = length // batch_size
        loss = 0
        enc.init_hidden(input_dim, gpu_id=gpu_id)
        dec.init_hidden(latent_dim, gpu_id=gpu_id)

        print("Processing file number {}".format(file))

        X = (dat).astype(np.float)
        X = Variable(torch.from_numpy(X)).cuda(gpu_id).float()

        # Forward
        z_mu, z_var = enc(X)
        z = sample_z(z_mu, z_var, X.size(0), latent_dim, gpu_id=gpu_id)
        X_sample = dec(z)

        # Loss
        recon_loss = F.mse_loss(X_sample, X)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        loss += 30. * recon_loss + kl_loss

        # Backward
        loss.backward()
        print(loss.data.cpu().numpy().flat[0])
        nn.utils.clip_grad_norm(enc.parameters(), 1)
        nn.utils.clip_grad_norm(dec.parameters(), 1)

        # Update
        solver_enc.step()
        solver_dec.step()

        # enc.reset_hidden()
        # dec.reset_hidden()
        loss = 0

        if int(file) % 10 == 0:
            torch.save(enc.cpu().state_dict(), 'classical_enc.mdl')
            torch.save(dec.cpu().state_dict(), 'classical_dec.mdl')

            enc = enc.cuda(gpu_id)
            dec = dec.cuda(gpu_id)
