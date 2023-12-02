import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch

from model import TrajectoryLSTM, AutoregressiveLSTM, VAEAutoencoder
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


def evaluate(ckpt_path="./ckpts/model_60000.pt", model_type='AutoregressiveLSTM'):
    """Evaluate the model on the validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path="val_data.pickle", num_workers=4)
    
    linear_layer = opt_linear(ckpt_path, model_type)
    linear_layer = torch.from_numpy(linear_layer).to(device)

    predict_ahead = 1
    hidden_size = 100
    hidden_size_param = 100
    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        hidden_size_param=hidden_size_param, 
        predict_ahead=predict_ahead
    ).to(device)

    # load the model
    if model_type == 'AutoregressiveLSTM':
        model = encoder
    elif model_type == 'VAEAutoencoder':
        decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, is_decoder=True).to(device)
        model = VAEAutoencoder(encoder, decoder, hidden_size).to(device)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Get predictions and ground truth parameters
    pred_params, gt_params, labels = [], [], []
    for i, (W, times, trajectories) in enumerate(dataloader):
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
    print("MAE (params) on the validation set: ", mae)

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


def visualize_trajectory(ckpt_path="./ckpts/model_1000.pt", idx=0, model_type='AutoregressiveLSTM'):
    """Analytically optimize a linear layer to map hidden vectors to parameters."""
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=4, shuffle=False)

    # Load model
    if model_type == 'AutoregressiveLSTM':
        predict_ahead = 30
        model = AutoregressiveLSTM(hidden_size=20, predict_ahead=predict_ahead).to(device)
    elif model_type == 'VAEAutoencoder':
        predict_ahead = 99
        encoder = AutoregressiveLSTM(hidden_size=20, predict_ahead=10).to(device)
        decoder = AutoregressiveLSTM(hidden_size=20, predict_ahead=predict_ahead, is_decoder=True).to(device)
        model = VAEAutoencoder(encoder, decoder, 20).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Get the hidden vectors and parameters
    for i, (W, times, trajectories) in enumerate(dataloader):        
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape

        # Run model
        data = trajectories.view(-1, time_size, state_size)
        input = data[:, :-1, :]
        targets = data[:, 1:, :]
        with torch.no_grad():
            model_outputs = model(input)
            predictions = model_outputs[0]
        break

    # Plot the trajectories
    gt_traj = targets[idx, :, :].detach().cpu().numpy()
    gt_times = times[0, 0, 1:].detach().cpu().numpy()

    if model_type == 'AutoregressiveLSTM':
        pred_traj = predictions[idx, time_size-predict_ahead-1, :].detach().cpu().numpy()
        pred_traj = np.concatenate((gt_traj[time_size-predict_ahead-2, :].reshape(1, -1), pred_traj), axis=0) # for continuation in visualization
        pred_times = times[0, 0, (time_size-predict_ahead-1):].detach().cpu().numpy()
        mae = np.mean(np.abs(pred_traj[1:] - gt_traj[-predict_ahead:]))
    elif model_type == 'VAEAutoencoder':
        pred_traj = predictions[idx].detach().cpu().numpy()
        pred_times = gt_times
        mae = np.mean(np.abs(pred_traj - gt_traj))

    print(mae)
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
    axs[0].set_title("m1 = {}, m2 = {}".format(*W[idx // 50, 2:].tolist()) + "\n" + "MAE = {:.4f}".format(mae))

    # Velocity subplot
    axs[1].plot(gt_times, gt_traj[:, 2], label="x1_dot (m1 velocity)")
    axs[1].plot(gt_times, gt_traj[:, 3], label="x2_dot (m2 velocity)")
    axs[1].plot(pred_times, pred_traj[:, 2], label="x1_dot (pred)")
    axs[1].plot(pred_times, pred_traj[:, 3], label="x2_dot (pred)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'./visualize_traj_{model_type}.png')
    

if __name__ == "__main__":
    # Evaluate parameters
    evaluate("./ckpts/model_10000.pt", 'AutoregressiveLSTM')
    # evaluate("./ckpts/vae_model_10000.pt", 'VAEAutoencoder')
    
    # Trajectories
    # visualize_trajectory("./ckpts/model_5000.pt", 100, 'AutoregressiveLSTM')
    # visualize_trajectory("./ckpts/vae_model_1000.pt", 100, 'VAEAutoencoder')
