import pdb
import json
import tqdm
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from info_nce import InfoNCE

from transformer import Transformer, TransformerDecoder
from dynamics import get_dataloader
from eval_transformer import evaluate_transformer, visualize_trajectory


def train_transformer(config=None):
    # Determine if we are doing hyperparameter search
    if config is None:
        wandb.init(config=config, project="contr_transformer", entity="contrastive-time-series")
        config = wandb.config
    
    log_to_wandb = config['log_to_wandb']
    # Determine if we are loggin
    if log_to_wandb:
        wandb.init(config=config, project="contr_transformer", entity="contrastive-time-series")
        
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
    mode = config['mode']
    
    # Load model
    if mode == 'encoder':
        model = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers).to(device)
    elif mode == 'decoder' or mode == 'decoder-contrastive':
        model = TransformerDecoder(d_input, d_model, d_linear, num_heads, dropout, num_layers).to(device)
    else:
        raise ValueError("Invalid mode")
    
    model.train()
    # model.load_state_dict(torch.load("./ckpts/framework1_best_2000.pt"))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-4)

    dataloader = get_dataloader(batch_size=32, data_path="./data/train_data.pickle", num_workers=8)
    mean, std = 0.0004, 0.6515

    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0.0
        
        for i, (W, times, trajectories) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get data
            trajectories = trajectories.to(device)
            trajectories = (trajectories - mean) / std
            batch_size, sample_size, time_size, state_size = trajectories.shape

            # Subsample
            indices = torch.randint(0, time_size, (batch_size, 2))
            batch_indices = torch.arange(25).view(-1, 1).expand(-1, 2)
            data = trajectories[batch_indices, indices, :]

            # Forward pass
            data = data.view(-1, time_size, state_size)
            emb = model(data)
            
            if mode == "encoder": # Compute contrastive loss
                loss = contrastive_loss(emb, batch_size)
            
            elif mode == "decoder": # Compute regression loss
                # Get predicted next steps
                emb = emb[:, 1:-1, :]
                out = model.pred_next_step(emb)
                # Compute regression loss
                loss = regression_loss(out, data)
            
            elif mode == "decoder-contrastive": # Compute contrastive loss + regression loss
                emb = emb[:, 1:-1, :]
                out = model.pred_next_step(emb)
                loss = contrastive_loss(emb, batch_size) + regression_loss(out, data)

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
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path) if log_to_wandb else None

            # Log metrics
            if mode == "encoder":
                train_mae, val_mae = evaluate_transformer(
                    model, ckpt_path=model_path
                )
                print("MAE (params) on the training set: ", train_mae.mean())
                wandb.log({"train_mae": train_mae.mean().item()}) if log_to_wandb else None
            elif mode == "decoder" or mode == "decoder-contrastive":
                traj_val_mae = visualize_trajectory(
                    model, ckpt_path=model_path,
                    val_data_path="./data/val_data.pickle",
                    visualize=False
                )
                print("MAE (trajectories) on the validation set: ", traj_val_mae)
                wandb.log({"val_traj_mae": traj_val_mae}) if log_to_wandb else None


def contrastive_loss(emb, batch_size):
    """Compute contrastive loss for a batch of embeddings from transformer encoder."""
    # Define loss function
    loss_fn = InfoNCE()
    
    # Separate positive and negative samples
    emb = emb[:, 0, :]
    emb = emb.view(batch_size, 2, -1)
    sample1 = emb[:, 0, :]
    sample2 = emb[:, 1, :]
    
    # Compute contrastive loss
    sample1 = sample1 / torch.norm(sample1, dim=1, keepdim=True)
    sample2 = sample2 / torch.norm(sample2, dim=1, keepdim=True)
    loss = loss_fn(sample1, sample2)
    return loss


def regression_loss(out, data):
    """Compute regression loss for a batch of embeddings from transformer decoder."""
    # Define loss function
    loss_fn = nn.MSELoss()
    # Get gt next steps
    y = data[:, 1:, :]
    # Compute regression loss
    loss = loss_fn(out, y)
    return loss


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
            'num_epochs': {'values': [3000]},
            'batch_size': {'values': [32]},
            'd_input': {'values': [4]},
            'd_model': {'values': [32, 64, 128, 256]},
            'd_linear': {'values': [16, 32, 64, 128, 256]},
            'num_heads': {'values': [4, 8]},
            'dropout': {'values': [0.1, 0.2, 0.4]},
            'num_layers': {'values': [2, 6, 10]},
            'mode': {'values': ['decoder']},
            'log_to_wandb': {'values': [True]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="contr_transformer", entity="contrastive-time-series")
    wandb.agent(sweep_id, train_transformer, count=20)


if __name__ == "__main__":
    # # Train single run
    # # Define hyperparameters
    # default_config = {
    #     'learning_rate': 0.00013,
    #     'num_epochs': 5000,
    #     'batch_size': 32,
    #     'd_input': 4,
    #     'd_model': 256,
    #     'd_linear': 64,
    #     'num_heads': 4,
    #     'dropout': 0.2,
    #     'num_layers': 4,
    #     'log_to_wandb': False,
    #     'mode': 'decoder-contrastive'
    # }

    # Or load from config file
    config_file = "./configs/transformer_encoder_config.json"
    with open(config_file, "r") as f:
        default_config = json.load(f)

    train_transformer(default_config)

    # # Hyperparameter search
    # transformer_hyperparam_search()