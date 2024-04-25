import pdb
import tqdm
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from info_nce import InfoNCE

from transformer import Transformer
from dynamics import get_dataloader
from eval_transformer import evaluate_transformer


def train_transformer(config=None):
    log_to_wandb = False

    # Determine if we are doing hyperparameter search
    if config is None:
        wandb.init(config=config, project="contr_transformer", entity="contrastive-time-series")
        config = wandb.config
        log_to_wandb = True
    
    # Determine if we are loggin
    if 'log_to_wandb' in config and config['log_to_wandb']:
        wandb.init(config=config, project="contr_transformer", entity="contrastive-time-series")
        log_to_wandb = True
        
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
    
    # Load model
    encoder = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers).to(device)
    encoder.train()
    # encoder.load_state_dict(torch.load("./ckpts/framework1_best_2000.pt"))
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-4)

    dataloader = get_dataloader(batch_size=32, data_path="./data/train_data.pickle", num_workers=8)
    loss_fn_contrastive = InfoNCE()

    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0.0
        
        for i, (W, times, trajectories) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get data
            trajectories = trajectories.to(device)
            batch_size, sample_size, time_size, state_size = trajectories.shape

            # Subsample
            indices = torch.randint(0, time_size, (batch_size, 2))
            batch_indices = torch.arange(25).view(-1, 1).expand(-1, 2)
            data = trajectories[batch_indices, indices, :]

            # Forward pass
            data = data.view(-1, time_size, state_size)
            embedding = encoder(data)

            # Contrastive loss
            embedding = embedding[:, 0, :]
            embedding = embedding.view(batch_size, 2, -1)

            # Compute loss - contrastive
            sample1 = embedding[:, 0, :]
            sample2 = embedding[:, 1, :]
            sample1 = sample1 / torch.norm(sample1, dim=1, keepdim=True)
            sample2 = sample2 / torch.norm(sample2, dim=1, keepdim=True)
            loss = loss_fn_contrastive(sample1, sample2)

            # # Compute loss - supervised
            # W = W.to(device)
            # pred_W = encoder.get_param(embedding)
            # loss = F.mse_loss(pred_W[:, 0, :], W[:, 2:]) + F.mse_loss(pred_W[:, 1, :], W[:, 2:])

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            total_loss += loss.item()

        # Log
        if epoch % 100 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))

            # Log metrics to wandb
            if log_to_wandb:
                wandb.log({"epoch": epoch, "total_loss": total_loss})
        
        # Save model
        if (epoch+1) % 100 == 0:
            model_path = f"./ckpts/model_{epoch+1}.pt"
            torch.save(encoder.state_dict(), model_path)
            train_mae, val_mae = evaluate_transformer(
                encoder, ckpt_path=model_path
            )
            print("MAE (params) on the training set: ", train_mae.mean())
            if log_to_wandb:
                wandb.save(model_path)  # Save model checkpoints to wandb
                wandb.log({"train_mae": train_mae.mean().item()})


def transformer_hyperparam_search():
    sweep_config = {
        'method': 'random', 
        'metric': {'name': 'loss', 'goal': 'minimize'},
        'parameters': {
            # 'learning_rate': {'distribution': 'log_uniform', 'min': int(np.floor(np.log(1e-4))), 'max': int(np.floor(np.log(1e-1)))},
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': np.log(1e-4),
                'max': np.log(1e-1)
            },
            'num_epochs': {'values': [3000]},
            'batch_size': {'values': [32]},
            'd_input': {'values': [4]},
            'd_model': {'values': [16, 32, 64, 128, 256]},
            'd_linear': {'values': [16, 32, 64, 128, 256]},
            'num_heads': {'values': [4, 8]},
            'dropout': {'values': [0.1, 0.2, 0.4]},
            'num_layers': {'values': [2, 6, 10]},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="contr_transformer", entity="contrastive-time-series")
    wandb.agent(sweep_id, train_transformer, count=20)


if __name__ == "__main__":
    # # Train single run
    # default_config = {
    #     'learning_rate': 1e-3,
    #     'num_epochs': 5000,
    #     'batch_size': 32,
    #     'd_input': 4,
    #     'd_model': 32,
    #     'd_linear': 64,
    #     'num_heads': 4,
    #     'dropout': 0.2,
    #     'num_layers': 6,
    #     'log_to_wandb': True,
    # }
    # train_transformer(default_config)

    # Hyperparameter search
    transformer_hyperparam_search()