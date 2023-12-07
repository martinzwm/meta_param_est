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
from eval import evaluate

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
    'learning_rate': 1e-2,
    'num_epochs': 2000,
    'predict_ahead': 1, # 1 for autoregressive, 99 for VAE
    'hidden_size': 100,
    'lambda_kl': 0,
    'lambda_contrastive': 0.0,
    'lambda_pred': 1.0,
    'is_vae': False,
    'bottleneck_size': 20,
    'num_layers': 8,
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
    
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_ahead = config['predict_ahead']
    hidden_size = config['hidden_size']
    lr = config['learning_rate']
    num_epochs = config['num_epochs']
    lambda_kl = config['lambda_kl']
    lambda_contrastive = config['lambda_contrastive']
    lambda_pred = config['lambda_pred']
    is_vae = config['is_vae']
    is_contrastive = config['lambda_contrastive'] > 0.0
    bottleneck_size = config['bottleneck_size']
    num_layers = config['num_layers']

    # Load model
    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        predict_ahead=predict_ahead,
        num_layers=num_layers,
    ).to(device)
    decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, num_layers=num_layers, is_decoder=True).to(device)
    vae = VAEAutoencoder(encoder, decoder, hidden_size, is_vae, bottleneck_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-4)
    
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=8)
    loss_fn_contrastive = InfoNCE()
    loss_fn_predictive = nn.MSELoss()
    loss_fn_reconstruct = nn.MSELoss()

    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss, total_loss_kl, total_loss_recon, total_loss_pred, total_loss_contrastive = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i, (W, times, trajectories) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get data
            trajectories = trajectories.to(device)
            batch_size, sample_size, time_size, state_size = trajectories.shape # batch_size = # of param. sets
            
            # Run VAE
            data = trajectories.view(-1, time_size, state_size)
            input = data[:, :-1, :]
            target = data[:, 1:, :]
            recon_x, encoder_out, mu, logvar, latent = vae(input)

            # Reconstruction Loss
            loss_recon = loss_fn_reconstruct(recon_x, target)
            
            # Encoder predictive Loss
            predictions_auto = encoder_out
            targets_auto = torch.zeros(batch_size * sample_size, time_size - predict_ahead, predict_ahead, state_size).to(device)
            for i in range(predict_ahead):
                targets_auto[:, :, i, :] = target[:, i:time_size-predict_ahead+i, :]
                
            loss_pred = loss_fn_predictive(predictions_auto, targets_auto)

            # KL Divergence Loss
            if is_vae:
                loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            else:
                loss_kl = torch.tensor(0.0)
    
            # Contrastive Loss
            if is_contrastive:
                hidden_vecs = latent.squeeze(0)
                hidden_vecs = hidden_vecs.view(batch_size, sample_size, -1)
                
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
            else:
                loss_contrastive = torch.tensor(0.0)
            
            # Total loss
            loss = loss_recon + lambda_pred * loss_pred + lambda_kl * loss_kl + lambda_contrastive * loss_contrastive
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            total_loss += loss.item()
            total_loss_recon += loss_recon.item()
            total_loss_pred += loss_pred.item()
            total_loss_kl += loss_kl.item()
            total_loss_contrastive += loss_contrastive.item()
        
        # Save model
        if epoch % 100 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))
            print("Reconstruction Loss: {}".format(total_loss_recon / len(dataloader)))
            print("Encoder predictive Loss: {}".format(total_loss_pred / len(dataloader)))
            print("KL Divergence: {}".format(total_loss_kl / len(dataloader)))
            print("Contrastive Loss: {}".format(total_loss_contrastive / len(dataloader)))
            
            # Log metrics to wandb
            if log_to_wandb:
                wandb.log({"epoch": epoch, "total_loss": total_loss, "recon_loss": total_loss_recon,
                        "predictive_loss": total_loss_pred, "kl_loss": total_loss_kl, "contrastive_loss": total_loss_contrastive})
        
        # Save model and evaluate
        if (epoch+1) % 1000 == 0:
            # Save
            model_path = f"./ckpts/vae_model_{epoch+1}.pt"
            torch.save(vae.state_dict(), model_path)
            # Evaluate
            params = {"hidden_size": hidden_size, "predict_ahead": predict_ahead, "is_vae": is_vae, "bottleneck_size": bottleneck_size, "num_layers": num_layers}
            param_mae = evaluate(ckpt_path=model_path, model_type='VAEAutoencoder', params=params)
            print("MAE (params) on the training set: ", param_mae)
            # Log
            if log_to_wandb:
                wandb.save(model_path)  # Save model checkpoints to wandb
                wandb.log({"param_mae": param_mae})


def train_contrastive(config=None):
    # Initialize wandb if config is not None
    log_to_wandb = False
    if config is None:
        wandb.init(config=config, project="vae_autoencoder", entity="contrastive-time-series")
        config = wandb.config
        log_to_wandb = True
    else:
        config = default_config
    
    # Hyperparameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = TrajectoryLSTM(hidden_size=10).to(device)
    lambda_traj = 10
    predict_ahead = config['predict_ahead']
    hidden_size = config['hidden_size']
    lr = config['learning_rate']
    num_epochs = config['num_epochs']
    
    # Load model
    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        predict_ahead=predict_ahead
    ).to(device)
    # encoder.load_state_dict(torch.load("./ckpts/model_10000.pt"))
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-4)

    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=8)
    loss_fn_contrastive = InfoNCE()
    loss_fn_predictive = nn.MSELoss()

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
            hidden_vecs = encoder.get_embedding(hidden_vecs)

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
            scheduler.step()

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
            torch.save(encoder.state_dict(), model_path)
            if log_to_wandb:
                wandb.save(model_path)  # Save model checkpoints to wandb


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


def vae_hyperparam_search():
    sweep_config = {
        'method': 'random', 
        'metric': {'name': 'total_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'distribution': 'log_uniform', 'min': int(np.floor(np.log(1e-3))), 'max': int(np.floor(np.log(1e-1)))},
            'num_epochs': {'values': [2000]},
            'hidden_size': {'values': [10, 50, 100, 150]},
            'lambda_kl': {'values': [0]},
            'lambda_contrastive': {'values': [0]},
            'lambda_pred': {'values': [1]},
            'predict_ahead': {'values': [1]},
            'is_vae': {'values': [False]},
            'framework': {'values': [4]},
            'bottleneck_size': {'values': [-1, 2, 10, 20]},
            'num_layers': {'values': [2, 4, 8]},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="vae_autoencoder", entity="contrastive-time-series")
    wandb.agent(sweep_id, train_vae_contrastive, count=20)
    

def lstm_hyperparam_search():
    sweep_config = {
        'method': 'random', 
        'metric': {'name': 'total_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'distribution': 'log_uniform', 'min': int(np.floor(np.log(1e-3))), 'max': int(np.floor(np.log(1e-1)))},
            'num_epochs': {'values': [2000]},
            'hidden_size': {'values': [10, 50, 100, 150]},
            'predict_ahead': {'values': [1, 10, 20]},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="lstm_autoregressive", entity="contrastive-time-series")
    wandb.agent(sweep_id, train_contrastive, count=5)
    

if __name__ == "__main__":
    # Train
    # train_contrastive(default_config)    
    # train_vae_contrastive(default_config)
    
    # # Sweep
    # lstm_hyperparam_search()
    vae_hyperparam_search()

    # # Evaluate on training set
    # for epoch in range(1000, 60001, 1000):
    #     ckpt_path = f"./ckpts/model_{epoch}.pt"
    #     print("Evaluating: ", ckpt_path)
    #     opt_linear(ckpt_path)