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
num_epoch = 20

batch_size = 200

enc = RecurrentEncoder(input_dim, latent_dim, lstm_layers).cuda(0)
dec = RecurrentDecoder(latent_dim, input_dim, lstm_layers).cuda(0)

solver_enc = optim.Adam(enc.parameters(), lr=1e-6)
solver_dec = optim.Adam(dec.parameters(), lr=1e-6)
print("At least this should probably run")

for it in range(num_epoch):
    files = npz.files
    shuffle(files)

    for file in files:
        dat = npz[file].T
        length, embed = dat.shape
        k = length // batch_size
        loss = 0
        enc.init_hidden(input_dim)
        dec.init_hidden(latent_dim)

        print("Processing file number {}".format(file))

        X = (dat).astype(np.float)
        X = Variable(torch.from_numpy(X)).cuda(0).float()

        # Forward
        z_mu, z_var = enc(X)
        z = sample_z(z_mu, z_var, X.size(0), latent_dim)
        X_sample = dec(z)

        # Loss
        recon_loss = F.mse_loss(X_sample, X, size_average=True)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        loss += recon_loss + kl_loss

        # Backward
        loss.backward()
        print(loss.data.cpu().numpy().flat[0])
        nn.utils.clip_grad_norm(enc.parameters(), 5)
        nn.utils.clip_grad_norm(dec.parameters(), 5)

        # Update
        solver_enc.step()
        solver_dec.step()

        enc.reset_hidden()
        dec.reset_hidden()
        loss = 0

        if int(file) % 200 == 0:
            torch.save(enc.cpu().state_dict(), 'enc.mdl')
            torch.save(dec.cpu().state_dict(), 'dec.mdl')

            enc = enc.cuda(0)
            dec = dec.cuda(0)
