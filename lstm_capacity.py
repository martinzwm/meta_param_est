import pdb
import tqdm
import numpy as np

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


def train_supervised():
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TrajectoryLSTM(hidden_size=100).to(device)
    # encoder = AutoregressiveLSTM(hidden_size=2, predict_ahead=1).to(device)
    encoder.load_state_dict(torch.load("./ckpts/exp/model_1000.pt"))
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-4)

    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=4)
    loss_fn = nn.MSELoss()

    num_epochs = 1000
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
            _, hidden_vecs = encoder(input)
            hidden_vecs = encoder.get_embedding(hidden_vecs)

            # Supervised learning to get loss
            hidden_vecs = hidden_vecs.view(batch_size * sample_size, -1)
            W = W[:, 2:].repeat_interleave(sample_size, dim=0).to(device)

            # Total loss
            loss = loss_fn(hidden_vecs, W)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            total_loss += loss.item()

        # Log
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(dataloader)))
        
        # Save model
        if (epoch+1) % 100 == 0:
            torch.save(encoder.state_dict(), f"./ckpts/exp/model_{epoch+1}.pt")


def eval_supervised():
    # Load data and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TrajectoryLSTM(hidden_size=100).to(device)
    encoder.load_state_dict(torch.load("./ckpts/exp/model_1000.pt"))
    dataloader = get_dataloader(batch_size=32, data_path="val_data.pickle", num_workers=4)

    # Evaluation loop - only 1 batch
    for i, (W, times, trajectories) in enumerate(dataloader):
        # Get data
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape
        
        # Run model
        data = trajectories.view(-1, time_size, state_size)
        input = data[:, :-1, :]
        _, hidden_vecs = encoder(input)
        hidden_vecs = encoder.get_embedding(hidden_vecs)

        # Supervised learning to get loss
        hidden_vecs = hidden_vecs.view(batch_size * sample_size, -1)
        W = W[:, 2:].repeat_interleave(sample_size, dim=0).to(device)
    
    # Calculate metrics
    mae = torch.mean(torch.abs(hidden_vecs - W))
    print("MAE: {}".format(mae))


if __name__ == "__main__":
    # # Train
    # train_supervised()

    # Evaluation
    eval_supervised()
