import pdb
import json
import tqdm
import numpy as np
import wandb

import torch
import torch.nn.functional as F
import torch.optim as optim

from autoencoder import Autoencoder, LinearAutoencoder, SimpleAutoencoder
from vorticity_dataset import get_image_dataloader
from eval_ae import eval_ae


def train_ae(config=None):
    # Determine if we are doing hyperparameter search
    if config is None:
        wandb.init(config=config, project="vorticity_autoencoder", entity="contrastive-time-series")
        config = wandb.config
    
    log_to_wandb = config['log_to_wandb']
    # Determine if we are loggin
    if log_to_wandb:
        wandb.init(config=config, project="vorticity_autoencoder", entity="contrastive-time-series")
        
    # Hyperparameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = config['learning_rate']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    dropout = config['dropout']
    mode = config['mode']
    latent_dim = config['latent_dim']

    # Model
    if mode == "Autoencoder":
        model = Autoencoder(dropout_rate=dropout).to(device)
    elif mode == "LinearAutoencoder":
        model = LinearAutoencoder(latent_dim=latent_dim).to(device)
    elif mode == "SimpleAutoencoder":
        model = SimpleAutoencoder(latent_dim=latent_dim, normalize=False).to(device)
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-4)

    dataloader = get_image_dataloader(batch_size=batch_size, data_path="./data/vorticity_train.pickle", num_workers=4)
    mean, std = 0, 0.905

    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0.0
        
        for i, image in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get data
            image = image.to(device)
            normalized_image = (image - mean) / std

            # Forward pass
            out = model(normalized_image)
            out = (out * std) + mean
            loss = F.mse_loss(out, image)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            total_loss += loss.item()

        # Log
        if (epoch+1) % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))

            # Log metrics to wandb
            if log_to_wandb:
                wandb.log({"epoch": epoch, "mse_train": total_loss / len(dataloader)})
        
        # Save model
        if (epoch+1) % 10 == 0:
            model_path = f"./ckpts/model_{epoch+1}.pt"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path) if log_to_wandb else None

            mse_val = eval_ae(model, model_path, "./data/vorticity_val.pickle")
            print("MSE on val set: {}".format(mse_val))
            wandb.log({"epoch": epoch, "mse_val": mse_val}) if log_to_wandb else None


def ae_hyperparam_search():
    sweep_config = {
        'method': 'random', 
        'metric': {'name': 'loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': np.log(1e-4),
                'max': np.log(1e-2)
            },
            'num_epochs': {'values': [300]},
            'batch_size': {'values': [32, 64, 128, 256]},
            'dropout': {'values': [0.1, 0.2, 0.4]},
            'log_to_wandb': {'values': [True]},
            'mode': {'values': ['SimpleAutoencoder']}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="vorticity_autoencoder", entity="contrastive-time-series")
    wandb.agent(sweep_id, train_ae, count=20)


if __name__ == "__main__":
    # Train single run
    # # Define hyperparameters
    # default_config = {
    #     'learning_rate': 0.0005,
    #     'num_epochs': 300,
    #     'batch_size': 128,
    #     'dropout': 0.2,
    #     'log_to_wandb': True,
    # }

    config_file = "./configs/ae_config.json"
    with open(config_file, "r") as f:
        default_config = json.load(f)
    train_ae(default_config)

    # Hyperparameter search
    # ae_hyperparam_search()