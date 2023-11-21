import pdb
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from info_nce import InfoNCE

from model import TrajectoryLSTM, AutoregressiveLSTM
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

def train_contrastive():
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = TrajectoryLSTM(hidden_size=10).to(device)
    lambda_traj = 10
    predict_ahead = 10
    encoder = AutoregressiveLSTM(hidden_size=20, predict_ahead=predict_ahead).to(device)
    # encoder.load_state_dict(torch.load("./ckpts/model_10000.pt"))
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

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

            # Select 2 trajectories from each set of parameters for contrastive learning
            indices = torch.randint(sample_size, (batch_size, 2), device=trajectories.device) # NOTE: could lead to duplicate samples
            sample1 = trajectories[torch.arange(batch_size), indices[:, 0]]
            sample2 = trajectories[torch.arange(batch_size), indices[:, 1]]

            # Run model
            data = torch.concat((sample1, sample2), dim=0)
            input = data[:, :-1, :]
            targets = data[:, 1:, :]
            predictions, hidden_vecs = encoder(input)
            
            # Predictive Loss
            # predictions_auto = predictions[:, :-(predict_ahead-1), :, :]
            predictions_auto = predictions
            targets_auto = torch.zeros(batch_size * 2, time_size - predict_ahead, predict_ahead, state_size).to(device)
            for i in range(predict_ahead):
                targets_auto[:, :, i, :] = targets[:, i:time_size-predict_ahead+i, :]

            loss_predictive = loss_fn_predictive(predictions_auto, targets_auto)

            # Contrastive loss
            hidden_vecs = hidden_vecs.squeeze(0)
            sample1 = hidden_vecs[:batch_size, :]
            sample2 = hidden_vecs[batch_size:, :]
            # Force top k dimension to be parameters
            sample1 = sample1[:, :10]
            sample2 = sample2[:, :10]
            sample1 = sample1 / torch.norm(sample1, dim=1, keepdim=True)
            sample2 = sample2 / torch.norm(sample2, dim=1, keepdim=True)
            loss_contrastive = loss_fn_contrastive(sample1, sample2)

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
        
        # Save model
        if (epoch+1) % 1000 == 0:
            torch.save(encoder.state_dict(), f"./ckpts/model_{epoch+1}.pt")


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
    print("MAE on the training set: ", mae)


def opt_linear(ckpt_path="./ckpts/model_9000.pt"):
    """Analytically optimize a linear layer to map hidden vectors to parameters."""
    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AutoregressiveLSTM(hidden_size=10, predict_ahead=10).to(device)
    encoder.load_state_dict(torch.load(ckpt_path))
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=4)

    # Get the hidden vectors and parameters
    y, y_hat = [], []
    for i, (W, times, trajectories) in enumerate(dataloader):        
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape

        # Run model
        data = trajectories.view(-1, time_size, state_size)
        input = data[:, :-1, :]
        with torch.no_grad():
            predictions, hidden_vecs = encoder(input)

        # Get the embeddings for all the trajectories in the batch
        hidden_vecs = hidden_vecs.view(batch_size * sample_size, -1)

        # Get the ground truth parameters
        W = W[:, 2:]
        W = W.repeat_interleave(sample_size, dim=0)

        y_hat.append(hidden_vecs)
        y.append(W)
    
    # Add columns of ones
    X = torch.cat(y_hat, dim=0).cpu()
    ones = torch.ones(X.shape[0], 1)
    X = torch.cat((X, ones), dim=1).cpu()
    Y = torch.cat(y, dim=0).cpu()

    # Optimize with analytical solution
    A_T_A_inv = torch.inverse(torch.mm(X.t(), X))
    A_T_B = torch.mm(X.t(), Y)
    linear_layer = torch.mm(A_T_A_inv, A_T_B)
    linear_layer = linear_layer.numpy()

    # Evaluate
    pred_params = np.matmul(X, linear_layer).numpy()
    gt_params = Y.numpy()
    mae = np.mean(np.abs(pred_params - gt_params), axis=0)
    print("MAE on the training set: ", mae)

    return linear_layer

if __name__ == "__main__":
    # # Train
    # train_contrastive()

    # Evaluat on a single checkpoint
    opt_linear("./ckpts/model_100000.pt")

    # # Evaluate on training set
    # for epoch in range(1000, 60001, 1000):
    #     ckpt_path = f"./ckpts/model_{epoch}.pt"
    #     print("Evaluating: ", ckpt_path)
    #     opt_linear(ckpt_path)