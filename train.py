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

# default config if no sweep
default_config = {
    'learning_rate': 1e-3,
    'hidden_size': 20,
    'lambda_kl': 0.1,
    'lambda_contrastive': 0.1,
}
    
def train_vae_contrastive(config=None):
    # Initialize wandb if config is not None
    log_to_wandb = False
    if config is None:
        wandb.init(config=config, project="vae_autoencoder", entity="contrastive-time-series")
        config = wandb.config
        log_to_wandb = True
    else:
        config = default_config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predict_ahead = 1
    hidden_size = config['hidden_size']
    lr = config['learning_rate']
    lambda_kl = config['lambda_kl']
    lambda_contrastive = config['lambda_contrastive']
    
    encoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=predict_ahead).to(device)
    decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, is_decoder=True).to(device)
    vae = VAEAutoencoder(encoder, decoder, hidden_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=4)
    loss_fn_contrastive = InfoNCE()
    loss_fn_reconstruct = nn.MSELoss()

    num_epochs = 10000
    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss, total_loss_kl, total_loss_recon, total_loss_contrastive = 0.0, 0.0, 0.0, 0.0
        
        for i, (W, times, trajectories) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get data
            trajectories = trajectories.to(device)
            batch_size, sample_size, time_size, state_size = trajectories.shape # batch_size = # of param. sets
            
            # Sample 2 trajectories from each set of parameters for contrastive learning
            indices = torch.randint(sample_size, (batch_size, 2), device=trajectories.device) # NOTE: could lead to duplicate samples
            sample1 = trajectories[torch.arange(batch_size), indices[:, 0]] # batch_size x seq_len x state_size
            sample2 = trajectories[torch.arange(batch_size), indices[:, 1]]
            data = torch.concat((sample1, sample2), dim=0)
            
            # Run VAE
            input = data[:, :-1, :]
            target = data[:, 1:, :]
            recon_x, mu, logvar, c_t = vae(input)
            
            # Reconstruction Loss
            loss_recon = loss_fn_reconstruct(recon_x, target)

            # KL Divergence Loss
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
            # Contrastive Loss
            hidden_vecs = c_t.squeeze(0)
            sample1 = hidden_vecs[:batch_size, :]
            sample2 = hidden_vecs[batch_size:, :]

            sample1 = F.normalize(sample1, p=2, dim=1)
            sample2 = F.normalize(sample2, p=2, dim=1)
            
            loss_contrastive = loss_fn_contrastive(sample1, sample2)
            
            # Total loss
            loss = loss_recon + lambda_kl * loss_kl + lambda_contrastive * loss_contrastive
            loss.backward()
            optimizer.step()

            # Logging
            total_loss += loss.item()
            total_loss_recon += loss_recon.item()
            total_loss_kl += loss_kl.item()
            total_loss_contrastive += loss_contrastive.item()
        
        # Save model
        if epoch % 100 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))
            print("Reconstruction Loss: {}".format(total_loss_recon / len(dataloader)))
            print("KL Divergence: {}".format(total_loss_kl / len(dataloader)))
            print("Contrastive Loss: {}".format(total_loss_contrastive / len(dataloader)))
            
            # Log metrics to wandb
            if log_to_wandb:
                wandb.log({"epoch": epoch, "total_loss": total_loss, "recon_loss": total_loss_recon,
                        "kl_loss": total_loss_kl, "contrastive_loss": total_loss_contrastive})
        
        # Save model
        if (epoch+1) % 1000 == 0:
            model_path = f"./ckpts/vae_model_{epoch+1}.pt"
            
            if log_to_wandb:
                wandb.save(model_path)  # Save model checkpoints to wandb
            else: 
                torch.save(vae.state_dict(), model_path)
            

def train_contrastive(config=None):
    # Initialize wandb if config is not None
    log_to_wandb = False
    if config is None:
        wandb.init(config=config, project="vae_autoencoder", entity="contrastive-time-series")
        config = wandb.config
        log_to_wandb = True
    else:
        config = default_config
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = TrajectoryLSTM(hidden_size=10).to(device)
    lambda_traj = 10
    predict_ahead = 30
    hidden_size = config['hidden_size']
    lr = config['learning_rate']
    encoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=predict_ahead).to(device)
    # encoder.load_state_dict(torch.load("./ckpts/model_10000.pt"))
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=4)
    loss_fn_contrastive = InfoNCE()
    loss_fn_predictive = nn.MSELoss()

    num_epochs = 10000
    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss, total_loss_contrastive, total_loss_predictive = 0.0, 0.0, 0.0
        
        for i, (W, times, trajectories) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get data
            trajectories = trajectories.to(device)
            batch_size, sample_size, time_size, state_size = trajectories.shape
            
            # Run model
            data = trajectories.view(-1, time_size, state_size)
            input = data[:, :-1, :]
            targets = data[:, 1:, :]
            predictions, hidden_vecs = encoder(input)

            # Predictive Loss
            predictions_auto = predictions
            targets_auto = torch.zeros(batch_size * sample_size, time_size - predict_ahead, predict_ahead, state_size).to(device)
            for i in range(predict_ahead):
                targets_auto[:, :, i, :] = targets[:, i:time_size-predict_ahead+i, :]
            
            loss_predictive = loss_fn_predictive(predictions_auto, targets_auto)

            # Contrastive loss
            hidden_vecs = hidden_vecs.squeeze(0)
            hidden_vecs = hidden_vecs.view(batch_size, sample_size, -1)

            # Sample sample_size number of times, this is a hyperparameter.
            loss_contrastive_list = []
            for i in range(10):
                indices = torch.randint(sample_size, (batch_size, 2), device=trajectories.device) # NOTE: could lead to duplicate samples
                sample1 = hidden_vecs[torch.arange(batch_size), indices[:, 0], :]
                sample2 = hidden_vecs[torch.arange(batch_size), indices[:, 1], :]

                # Force top k dimension to be corresponds to parameters
                sample1 = sample1[:, :10]
                sample2 = sample2[:, :10]
                sample1 = sample1 / torch.norm(sample1, dim=1, keepdim=True)
                sample2 = sample2 / torch.norm(sample2, dim=1, keepdim=True)
                loss_contrastive_list.append(loss_fn_contrastive(sample1, sample2))
            loss_contrastive = torch.mean(torch.stack(loss_contrastive_list))

            # Total loss
            loss = lambda_traj * loss_predictive + loss_contrastive
            loss.backward()
            optimizer.step()

            # Logging
            total_loss += loss.item()
            total_loss_contrastive += loss_contrastive.item()
            total_loss_predictive += loss_predictive.item()

        # Log
        if epoch % 100 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))
            print("Predictive Loss: {}".format(total_loss_predictive / len(dataloader)))
            print("Contrastive Loss: {}".format(total_loss_contrastive / len(dataloader)))
            # print("Sample 1: {}".format(sample1[:2]))
            # print("Sample 2: {}".format(sample2[:2]))

            # Log metrics to wandb
            if log_to_wandb:
                wandb.log({"epoch": epoch, "total_loss": total_loss, "predictive_loss": total_loss_predictive,
                        "contrastive_loss": total_loss_contrastive})
        
        # Save model
        if (epoch+1) % 1000 == 0:
            model_path = f"./ckpts/model_{epoch+1}.pt"
            if log_to_wandb:
                wandb.save(model_path)  # Save model checkpoints to wandb
            else: 
                torch.save(encoder.state_dict(), model_path)


def train_linear(ckpt_path="./ckpts/model_10000.pt", verbose=False):
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TrajectoryLSTM(hidden_size=10).to(device)
    encoder.load_state_dict(torch.load(ckpt_path))
    optimizer = optim.Adam(encoder.parameters(), lr=2e-3)
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle")
    loss_fn = nn.MSELoss()

    num_epochs = 10000
    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0.0
        
        for i, (W, times, trajectories) in enumerate(dataloader):
            optimizer.zero_grad()
            
            trajectories = trajectories.to(device)
            batch_size, sample_size, time_size, state_size = trajectories.shape

            # Run model
            data = trajectories.view(-1, time_size, state_size)
            input = data[:, :-1, :]
            with torch.no_grad():
                predictions, hidden_vecs = encoder(input)
            hidden_vecs = encoder.get_embedding(hidden_vecs)

            # Get the embeddings for all the trajectories in the batch
            hidden_vecs = hidden_vecs.view(batch_size, sample_size, -1)

            # Select 1 trajectories from each set of parameters for contrastive learning
            sample = []
            for i in range(batch_size):
                idx1 = np.random.choice(sample_size, 1, replace=False)
                sample.append(hidden_vecs[i, idx1, :])
            sample = torch.stack(sample)
            sample.squeeze_(1)

            # Compute loss
            loss = loss_fn(sample, W[:, 2:].to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Log
        if epoch % 100 == 0 and verbose:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))
        
        # Save model
        if (epoch+1) % 1000 == 0:
            torch.save(encoder.state_dict(), f"./ckpts/model_linear_{epoch+1}.pt")
        
    # Evaluate
    gt_params = W[:, 2:].numpy()
    pred_params = sample.detach().cpu().numpy()
    mae = np.mean(np.abs(pred_params - gt_params), axis=0)
    print("MAE (params) on the training set: ", mae)


def get_hidden_vectors_and_params(model, dataloader, device):
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
    # Optimize with analytical solution
    A_T_A_inv = torch.inverse(torch.mm(X.t(), X))
    A_T_B = torch.mm(X.t(), Y)
    linear_layer = torch.mm(A_T_A_inv, A_T_B)

    return linear_layer.numpy()


def opt_linear(ckpt_path, model_type='AutoregressiveLSTM'):
    """
    Evaluate the model on the validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=4)

    # Load the model
    if model_type == 'AutoregressiveLSTM':
        model = AutoregressiveLSTM(hidden_size=20, predict_ahead=20).to(device)
    elif model_type == 'VAEAutoencoder':
        encoder = AutoregressiveLSTM(hidden_size=20, predict_ahead=10).to(device)
        decoder = AutoregressiveLSTM(hidden_size=20, predict_ahead=99, is_decoder=True).to(device)
        model = VAEAutoencoder(encoder, decoder, 20).to(device)
          
    model.load_state_dict(torch.load(ckpt_path, map_location = device))
    model.eval()

    # Extract hidden vectors and parameters
    hidden_vecs, gt_params = get_hidden_vectors_and_params(model, dataloader, device) # hidden_vecs here has bias column

    # Solve for the linear system
    linear_layer = solve(hidden_vecs, gt_params)

    # Evaluate using the linear_layer
    pred_params = np.matmul(hidden_vecs, linear_layer).numpy()
    mae = np.mean(np.abs(pred_params - gt_params.numpy()), axis=0)
    print("MAE (params) on the training set: ", mae)

    return linear_layer


def vae_hyperparam_search():
    sweep_config = {
        'method': 'bayes', 
        'metric': {'name': 'total_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'distribution': 'log_uniform', 'min': -10, 'max': -4},
            'hidden_size': {'values': [20, 40, 60]},
            'lambda_kl': {'values': [0.05, 0.1]},
            'lambda_contrastive': {'values': [0.1, 1]}}
        }
    sweep_id = wandb.sweep(sweep_config, project="vae_autoencoder", entity="contrastive-time-series")
    wandb.agent(sweep_id, train_vae_contrastive)
    

def lstm_hyperparam_search():
    sweep_config = {
        'method': 'bayes', 
        'metric': {'name': 'total_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'distribution': 'log_uniform', 'min': -12, 'max': -6},
            'hidden_size': {'values': [20, 40, 60]}}
        }
    sweep_id = wandb.sweep(sweep_config, project="lstm_autoregressive", entity="contrastive-time-series")
    wandb.agent(sweep_id, train_contrastive)
    
if __name__ == "__main__":
    # # Train
    train_contrastive(default_config)    
    # train_vae_contrastive(default_config)
    
    # # Sweep
    # lstm_hyperparam_search()
    # vae_hyperparam_search()
    

    # Evaluat on a single checkpoint
    # opt_linear("./ckpts/model_100000.pt")
    # opt_linear("./ckpts/vae_model_1000.pt", 'VAEAutoencoder')

    # # Evaluate on training set
    # for epoch in range(1000, 60001, 1000):
    #     ckpt_path = f"./ckpts/model_{epoch}.pt"
    #     print("Evaluating: ", ckpt_path)
    #     opt_linear(ckpt_path)
    
