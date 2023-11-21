import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch

from model import TrajectoryLSTM, AutoregressiveLSTM
from dynamics import get_dataloader
from train import opt_linear

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


def evaluate(ckpt_path="./ckpts/model_60000.pt"):
    """Evaluate the model on the validation set."""
    # Load model and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = TrajectoryLSTM(hidden_size=10).to(device)
    encoder = AutoregressiveLSTM(hidden_size=10, predict_ahead=10).to(device)
    encoder.load_state_dict(torch.load(ckpt_path))
    encoder.eval()

    dataloader = get_dataloader(batch_size=32, data_path="val_data.pickle", num_workers=4)
    linear_layer = opt_linear(ckpt_path)
    linear_layer = torch.from_numpy(linear_layer).to(device)

    # Get predictions and ground truth parameters
    pred_params, gt_params, labels = [], [], []
    for i, (W, times, trajectories) in enumerate(dataloader):
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape

        # Run model
        data = trajectories.view(-1, time_size, state_size)
        input = data[:, :-1, :]
        with torch.no_grad():
            predictions, hidden_vecs = encoder(input)
        
        # Add a bias term to the hidden vectors
        hidden_vecs = hidden_vecs.squeeze(0)
        ones = torch.ones(hidden_vecs.shape[0], 1).to(device)
        X = torch.cat((hidden_vecs, ones), dim=1)
        pred_W = torch.mm(X, linear_layer).unsqueeze(0)

        # Get the embeddings for all the trajectories in the batch
        pred_W = pred_W.view(batch_size * sample_size, -1)
        
        # Get the ground truth parameters
        W = W[:, 2:]
        gt_W = W.repeat_interleave(sample_size, dim=0)
        labels.append(gt_W.detach().cpu().numpy())

        # Save the embeddings and labels
        pred_params.append(pred_W.detach().cpu().numpy())
        gt_params.append(gt_W.detach().cpu().numpy())

    # Convert to numpy arrays
    pred_params = np.concatenate(pred_params, axis=0)
    gt_params = np.concatenate(gt_params, axis=0)
    labels = np.concatenate(labels, axis=0)
    string_labels = [",".join(map(str, row)) for row in labels] # Only take the m1 and m2 values

    # MSE
    mae = np.mean(np.abs(pred_params - gt_params), axis=0)
    print("MAE: ", mae)

    visualize_params_with_labels(pred_params, gt_params, string_labels)


def visualize_params_with_labels(pred_params, gt_params, labels):
    """Visualize the ground truth and predicted parameters."""
    # Ensure data is numpy array
    pred_params = np.array(pred_params)
    gt_params = np.array(gt_params)
    labels = np.array(labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normalize values for colormap scaling
    gt_norm1 = plt.Normalize(gt_params[:,0].min(), gt_params[:,0].max())
    gt_norm2 = plt.Normalize(gt_params[:,1].min(), gt_params[:,1].max())

    for i in range(len(gt_params)):
        # Use red colormap for first value and blue colormap for second value
        red_colormap = cm.Reds(gt_norm1(gt_params[i, 0]))
        blue_colormap = cm.Blues(gt_norm2(gt_params[i, 1]))
        
        # Combine the two colormaps to produce a single color for the point
        combined_color = np.clip([red_colormap[j] + blue_colormap[j] for j in range(3)], 0, 1)
        
        ax1.scatter(gt_params[i, 0], gt_params[i, 1], color=combined_color)
        # ax1.xlabel("m1"); ax1.ylabel("m2")
        ax1.text(gt_params[i, 0], gt_params[i, 1], str(labels[i]), fontsize=8)

        ax2.scatter(pred_params[i, 0], pred_params[i, 1], color=combined_color)
        # ax2.xlabel("m1"); ax2.ylabel("m2")
        ax2.text(pred_params[i, 0], pred_params[i, 1], str(labels[i]), fontsize=8)

    ax1.set_title('Ground Truth Parameters')
    ax2.set_title('Predicted Parameters')
    
    plt.tight_layout()
    plt.savefig('./params.png')


def visualize_trajectory(ckpt_path="./ckpts/model_1000.pt", idx=0):
    """Analytically optimize a linear layer to map hidden vectors to parameters."""
    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AutoregressiveLSTM(hidden_size=10, predict_ahead=10).to(device)
    encoder.load_state_dict(torch.load(ckpt_path))
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=4)

    # Get the hidden vectors and parameters
    for i, (W, times, trajectories) in enumerate(dataloader):        
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape

        # Run model
        data = trajectories.view(-1, time_size, state_size)
        input = data[:, :-1, :]
        targets = data[:, 1:, :]
        with torch.no_grad():
            predictions, hidden_vecs = encoder(input)
        break

    # Plot the trajectories
    gt_traj = targets[idx, :, :].detach().cpu().numpy()
    gt_times = times[0, 0, 1:].detach().cpu().numpy()
    pred_traj = predictions[idx, 89, :].detach().cpu().numpy()
    pred_traj = np.concatenate((gt_traj[88, :].reshape(1, -1), pred_traj), axis=0) # for continuation in visualization
    pred_times = times[0, 0, 89:].detach().cpu().numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Displacement subplot
    axs[0].plot(gt_times, gt_traj[:, 0], label="x1 (m1 displacement)")
    axs[0].plot(gt_times, gt_traj[:, 1], label="x2 (m2 displacement)")
    axs[0].plot(pred_times, pred_traj[:, 0], label="x1 (pred)")
    axs[0].plot(pred_times, pred_traj[:, 1], label="x2 (pred)")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Displacement")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Displacement vs. Time")

    # Velocity subplot
    axs[1].plot(gt_times, gt_traj[:, 2], label="x1_dot (m1 velocity)")
    axs[1].plot(gt_times, gt_traj[:, 3], label="x2_dot (m2 velocity)")
    axs[1].plot(pred_times, pred_traj[:, 2], label="x1_dot (pred)")
    axs[1].plot(pred_times, pred_traj[:, 3], label="x2_dot (pred)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("Velocity vs. Time")

    plt.tight_layout()
    plt.savefig('./visualize_traj.png')
    

if __name__ == "__main__":
    # evaluate(ckpt_path="./ckpts/model_60000.pt") # parameter
    visualize_trajectory(ckpt_path="./ckpts/model_60000.pt", idx=0) # trajectory