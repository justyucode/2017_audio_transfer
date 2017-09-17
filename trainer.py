import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from model import RecurrentEncoder, RecurrentDecoder, sample_z


# =============================== TRAINING ====================================
npz = np.load("training_data.npz")

input_dim = 1000
latent_dim = 100
lstm_layers = 2
num_epoch = 2

batch_size = 100

enc = RecurrentEncoder(input_dim + latent_dim, latent_dim, lstm_layers).cuda(0)
dec = RecurrentDecoder(latent_dim, input_dim, lstm_layers).cuda(0)
enc.init_hidden(input_dim)
dec.init_hidden(latent_dim)

solver_enc = optim.Adam(enc.parameters(), lr=1e-5)
solver_dec = optim.Adam(dec.parameters(), lr=1e-5)
print("At least this should probably run")

for it in range(num_epoch):
    for file in npz.files:
        dat = npz[file]
        length, embed = dat.shape
        k = length // batch_size
        loss = 0

        for i in range(k+1):
            if i != k:
                X = dat[i * batch_size: (i + 1) * batch_size, :]
            else:
                X = dat[i * batch_size:, :]

            for index in range(k):
                X = X[index:index+1]
                X = Variable(torch.from_numpy(X)).cuda(0)
                cat_X = torch.cat([X, dec.cell_state.squeeze()[-1:]], 1)

                # Forward
                z_mu, z_var = enc(cat_X)
                z = sample_z(z_mu, z_var, 1, latent_dim)
                X_sample = dec(z)

                # Loss
                recon_loss = F.mse_loss(X_sample, X, size_average=True)
                kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
                loss += recon_loss + kl_loss

            # Backward
            loss.backward()
            nn.utils.clip_grad_norm(enc.parameters(), 1)
            nn.utils.clip_grad_norm(dec.parameters(), 1)

            # Update
            solver_enc.step()
            solver_dec.step()

            enc.reset_hidden()
            dec.reset_hidden()
