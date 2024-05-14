import pdb
import json
import tqdm
import wandb
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(0, './../')
from model.transformer import Transformer, TransformerDecoder, EncoderDecoder
from train_transformer import contrastive_loss, regression_loss

from autoencoder import LinearAutoencoder, DummyAutoencoder
from vorticity_dataset import get_video_dataloader
from eval_transformer_vorticity import eval_transformer, eval_embeddings


def sample_subtrajectories(trajectories, L):
    """Sample 2 nonoverlaping trajectory of given length L from the given trajectories."""
    B, T, C, H, W = trajectories.shape
    sub_traj_1 = torch.zeros(B, L, C, H, W, dtype=trajectories.dtype, device=trajectories.device)
    sub_traj_2 = torch.zeros(B, L, C, H, W, dtype=trajectories.dtype, device=trajectories.device)

    for b in range(B):
        # Starting index for the first sub-trajectory
        max_start_1 = T - 2 * L        
        start_1 = torch.randint(0, max_start_1 + 1, (1,))
        
        # Starting index for the second sub-trajectory
        min_start_2 = int(start_1) + L
        max_start_2 = T - L
        start_2 = torch.randint(min_start_2, max_start_2 + 1, (1,))
        
        # Extract the sub-trajectories
        sub_traj_1[b] = trajectories[b, start_1:start_1 + L]
        sub_traj_2[b] = trajectories[b, start_2:start_2 + L]

    return sub_traj_1, sub_traj_2


def train_transformer(config=None):
    # Determine if we are doing hyperparameter search
    if config is None:
        wandb.init(config=config, project="vorticity", entity="contrastive-time-series")
        config = wandb.config
    
    log_to_wandb = config['log_to_wandb']
    # Determine if we are loggin
    if log_to_wandb:
        wandb.init(config=config, project="vorticity", entity="contrastive-time-series")
        
    # Hyperparameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = config['learning_rate']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    d_input = config['d_input']
    d_model = config['d_model']
    d_linear = config['d_linear']
    num_heads = config['num_heads']
    dropout = config['dropout']
    num_layers = config['num_layers']

    # Load autoencoder model
    autoencoder = LinearAutoencoder(latent_dim=d_input).to(device)
    autoencoder.load_state_dict(torch.load("./ckpts/ae_linear_best.pt"))
    # autoencoder = DummyAutoencoder().to(device)
    
    # Load transformer model
    encoder = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers).to(device)
    decoder = TransformerDecoder(d_input, d_model, d_linear, num_heads, dropout, num_layers).to(device)
    model = EncoderDecoder(encoder, decoder).to(device)
    optimizer = optim.Adam(chain(model.parameters(), autoencoder.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-4)

    # Get dataloader
    dataloader = get_video_dataloader(batch_size=32, data_path="./data/vorticity_train.pickle", num_workers=8)
    mean, std = 0, 0.905

    model.train()
    autoencoder.train()
    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0.0
        
        for i, (trajectories, vorticities, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            B, T, C, H, W = trajectories.shape
            L = 40

            # Randomly sample two time points
            sub_traj_1, sub_traj_2 = sample_subtrajectories(trajectories, L)
            gt_traj = torch.cat([sub_traj_1, sub_traj_2], dim=0).to(device)
            data = gt_traj.view(2*B, 40, -1)
            data = (data - mean) / std

            # Extract features from autoencoder.encoder
            with torch.no_grad():
                features = autoencoder.encoder(data)
            
            # Compute loss
            loss_contr, loss_next_step, loss_traj = 0, 0, 0
            idx = 20
            enc_emb, dec_emb = model(features, idx)

            # Compute contrastive loss
            loss_contr = contrastive_loss(enc_emb, B)

            # Compute regression loss on predicted next steps
            emb = dec_emb[:, 1:-1, :]
            out = model.decoder.pred_next_step(emb)
            with torch.no_grad():
                out = autoencoder.decoder(out)
            out = out.view(2*B, -1, C, H, W)
            out = (out * std) + mean
            loss_next_step = regression_loss(out, gt_traj[:, idx:, :])

            # Compute regression loss on predict full trajectory
            out = model.decoder.generate(features[:, idx-5:idx, :], L-idx-1, enc_emb)
            out = out[:, 5:, :] # exclude the inital point
            with torch.no_grad():
                out = autoencoder.decoder(out)
            out = out.view(2*B, -1, C, H, W)
            loss_traj = regression_loss(out, gt_traj[:, idx:, :])
            
            loss = 0.01 * loss_contr + loss_next_step + loss_traj

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            total_loss += loss.item()

        # Log
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))

            # Log metrics to wandb
            if log_to_wandb:
                wandb.log({
                    "epoch": epoch, 
                    "loss": total_loss / len(dataloader),
                    "loss_contr": loss_contr.item(),
                    "loss_next_step": loss_next_step.item(),
                    "loss_traj": loss_traj.item()
                })
        
        # Save model
        if (epoch+1) % 100 == 0:
            model_path = f"./ckpts/transformer_{epoch+1}.pt"
            torch.save(model.state_dict(), model_path)
            # wandb.save(model_path) if log_to_wandb else None
            autoencoder_path = f"./ckpts/ae_{epoch+1}.pt"
            torch.save(autoencoder.state_dict(), autoencoder_path)
            # wandb.save(autoencoder_path) if log_to_wandb else None

            # Evaluate
            mse_traj_train = eval_transformer(
                model, 
                autoencoder,
                model_path,
                autoencoder_path,
                './data/vorticity_train.pickle', 
                False,
            )

            mse_traj_val = eval_transformer(
                model, 
                autoencoder,
                model_path,
                autoencoder_path,
                './data/vorticity_val.pickle', 
                False,
            )

            mae_param_train, mae_param_val = eval_embeddings(
                model, 
                autoencoder,
                model_path,
                autoencoder_path,
                './data/vorticity_train.pickle',
                './data/vorticity_val.pickle', 
                False,
            )

            print(f"Val Emb MAE: {mae_param_val}, Train Emb MAE: {mae_param_train}")
            print(f"Val Traj MSE: {mse_traj_val}, Train Traj MSE: {mse_traj_train}")

            wandb.log({
                "val_emb_mae": mae_param_val,
                "train_emb_mae": mae_param_train,
                "val_traj_mse": mse_traj_val,
                "train_traj_mse": mse_traj_train,
            }) if log_to_wandb else None


def transformer_hyperparam_search():
    sweep_config = {
        'method': 'random', 
        'metric': {'name': 'loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': np.log(1e-4),
                'max': np.log(1e-2)
            },
            'num_epochs': {'values': [1000]},
            'batch_size': {'values': [32]},
            'd_input': {'values': [512]},
            'd_model': {'values': [32, 64, 128, 256]},
            'd_linear': {'values': [16, 32, 64, 128, 256]},
            'num_heads': {'values': [4, 8]},
            'dropout': {'values': [0.1, 0.2, 0.4]},
            'num_layers': {'values': [2, 6, 10]},
            'log_to_wandb': {'values': [True]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="vorticity_transformer", entity="contrastive-time-series")
    wandb.agent(sweep_id, train_transformer, count=20)


if __name__ == "__main__":
    # Train single run
    config_file = "./configs/transformer_config.json"
    with open(config_file, "r") as f:
        default_config = json.load(f)

    train_transformer(default_config)

    # # Hyperparameter search
    # transformer_hyperparam_search()