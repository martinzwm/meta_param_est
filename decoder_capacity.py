import pdb
import tqdm
import numpy as np
import wandb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from info_nce import InfoNCE

from model import TrajectoryLSTM, AutoregressiveLSTM, VAEAutoencoder
from dynamics import get_dataloader

# seed
seed_value = 42
import random
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train_vae_contrastive(config=None):
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_ahead = config['predict_ahead']
    hidden_size = config['hidden_size']
    lr = config['learning_rate']
    num_epochs = config['num_epochs']
    lambda_kl = config['lambda_kl']

    # Load model
    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        predict_ahead=predict_ahead
    ).to(device)
    
    # # Load best checkpoint from framework 1
    # ckpt_path = "./ckpts/framework1_best.pt"
    # encoder.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Load the rest of the model
    decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, is_decoder=True).to(device)
    vae = VAEAutoencoder(encoder, decoder, hidden_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-4)
    
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=8)
    loss_fn_reconstruct = nn.MSELoss()

    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss, total_loss_kl, total_loss_recon, total_loss_contrastive = 0.0, 0.0, 0.0, 0.0
        
        for i, (W, times, trajectories) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get data
            trajectories = trajectories.to(device)
            batch_size, sample_size, time_size, state_size = trajectories.shape # batch_size = # of param. sets
            data = trajectories.view(-1, time_size, state_size)
            
            # Run VAE
            input = data[:, :-1, :]
            target = data[:, 1:, :]
            recon_x, mu, logvar, c_t = vae(input)
            
            # Reconstruction Loss
            loss_recon = loss_fn_reconstruct(recon_x, target)

            # KL Divergence Loss
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss
            loss = loss_recon + lambda_kl * loss_kl
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            total_loss += loss.item()
            total_loss_recon += loss_recon.item()
            total_loss_kl += loss_kl.item()
        
        # Save model
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))
            print("Reconstruction Loss: {}, KL Loss: {}".format(total_loss_recon / len(dataloader), total_loss_kl / len(dataloader)))
            
        # Save model
        if (epoch+1) % 100 == 0:
            model_path = f"./ckpts/vae_model_{epoch+1}.pt"
            torch.save(vae.state_dict(), model_path)


def get_hidden_vectors_and_params(model, dataloader, device, model_type="AutoregressiveLSTM"):
    """
    Extract hidden vectors (a column of ones is added as bias) and ground truth parameters for a given model.
    """
    y, y_hat = [], []
    for _, (W, _, trajectories) in enumerate(dataloader):        
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape

        # Run model
        data = trajectories.view(-1, time_size, state_size)
        input = data[:, :-1, :]

        with torch.no_grad():
            model_outputs = model(input)
            hidden_vecs = model_outputs[-1] # the last entry is the hidden_vecs
            if model_type == "AutoregressiveLSTM":
                hidden_vecs = model.get_embedding(hidden_vecs)

        # Reshape and store hidden vectors and ground truth parameters
        hidden_vecs = hidden_vecs.view(batch_size * sample_size, -1)
        W = W[:, 2:].repeat_interleave(sample_size, dim=0)

        y_hat.append(hidden_vecs)
        y.append(W)
    
    # Add columns of ones for bias term
    hidden_vecs = torch.cat(y_hat, dim=0).cpu()
    ones = torch.ones(hidden_vecs.shape[0], 1)
    hidden_vecs_with_bias = torch.cat((hidden_vecs, ones), dim=1).cpu()
    gt_params = torch.cat(y, dim=0).cpu()

    return hidden_vecs_with_bias, gt_params

def solve(X, Y):
    """
    Optimize a linear layer to map hidden vectors to parameters.
    """
    linear_layer = torch.linalg.lstsq(X, Y).solution
    return linear_layer.numpy()


def opt_linear(ckpt_path, model_type='AutoregressiveLSTM', params=None):
    """
    Evaluate the model on the validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=4)

    hidden_size = params["hidden_size"] if params is not None else 100
    predict_ahead = params["predict_ahead"] if params is not None else 1
    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        predict_ahead=predict_ahead
    ).to(device)

    # Load the model
    if model_type == 'AutoregressiveLSTM':
        model = encoder
    elif model_type == 'VAEAutoencoder':
        decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, is_decoder=True).to(device)
        model = VAEAutoencoder(encoder, decoder, hidden_size).to(device)
          
    model.load_state_dict(torch.load(ckpt_path, map_location = device))
    model.eval()

    # Extract hidden vectors and parameters
    hidden_vecs, gt_params = get_hidden_vectors_and_params(model, dataloader, device, model_type) # hidden_vecs here has bias column

    # Solve for the linear system
    linear_layer = solve(hidden_vecs, gt_params)

    # Evaluate using the linear_layer
    pred_params = np.matmul(hidden_vecs, linear_layer).numpy()
    mae = np.mean(np.abs(pred_params - gt_params.numpy()), axis=0)
    print("MAE (params) on the training set: ", mae)
    return linear_layer


default_config = {
    'learning_rate': 1e-2,
    'num_epochs': 2000,
    'predict_ahead': 1, # 1 for autoregressive, 99 for VAE
    'hidden_size': 100,
    'lambda_kl': 0.0,
}

if __name__ == "__main__":
    # Train
    train_vae_contrastive(default_config)

    # # Evaluat on a single checkpoint
    # opt_linear("./ckpts/vae_model_1000.pt", 'VAEAutoencoder')