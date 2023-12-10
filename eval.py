import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch

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


def get_hidden_vectors_and_params(model, dataloader, device, model_type="AutoregressiveLSTM"):
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
            if model_type == "AutoregressiveLSTM":
                hidden_vecs = model.get_embedding(hidden_vecs)

        # Reshape and store hidden vectors and ground truth parameters
        hidden_vecs = hidden_vecs.mean(dim=0)
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
    linear_layer = torch.linalg.lstsq(X, Y).solution
    return linear_layer.numpy()


def opt_linear(ckpt_path, train_data_path="train_data.pickle", model_type='AutoregressiveLSTM', params=None):
    """
    Evaluate the model on the validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path=train_data_path, num_workers=4)

    hidden_size = params["hidden_size"] if params is not None else 100
    predict_ahead = params["predict_ahead"] if params is not None else 1
    bottleneck_size = params["bottleneck_size"] if params is not None else -1
    num_layers = params["num_layers"] if params is not None else 1
    embedding_out = params["embedding_out"] if params is not None else -1

    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        predict_ahead=predict_ahead,
        num_layers=num_layers,
        embedding_out=embedding_out
    ).to(device)

    # Load the model
    if model_type == 'AutoregressiveLSTM':
        model = encoder
    elif model_type == 'VAEAutoencoder':
        is_vae = params["is_vae"] if params is not None else False
        decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, num_layers=num_layers, is_decoder=True).to(device)
        model = VAEAutoencoder(encoder, decoder, hidden_size, is_vae, bottleneck_size).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location = device))
    model.eval()

    # Extract hidden vectors and parameters
    hidden_vecs, gt_params = get_hidden_vectors_and_params(model, dataloader, device, model_type) # hidden_vecs here has bias column

    # Solve for the linear system
    linear_layer = solve(hidden_vecs, gt_params)

    # Evaluate using the linear_layer
    pred_params = np.matmul(hidden_vecs, linear_layer).numpy()
    mae = np.mean(np.abs(pred_params - gt_params.numpy()), axis=0)
    print("MAE (params) on the training set: ", mae)
    return linear_layer


def evaluate(
    ckpt_path="./ckpts/model_60000.pt", 
    train_data_path="train_data.pickle",
    val_data_path="val_data.pickle",
    model_type='AutoregressiveLSTM', 
    params=None
):
    """Evaluate the model on the validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path=val_data_path, num_workers=4, shuffle=False)
    
    linear_layer = opt_linear(ckpt_path, train_data_path, model_type, params)
    linear_layer = torch.from_numpy(linear_layer).to(device)

    hidden_size = params["hidden_size"] if params is not None else 100
    predict_ahead = params["predict_ahead"] if params is not None else 1
    bottleneck_size = params["bottleneck_size"] if params is not None else -1
    num_layers = params["num_layers"] if params is not None else 1
    embedding_out = params["embedding_out"] if params is not None else -1

    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        predict_ahead=predict_ahead,
        num_layers=num_layers,
        embedding_out=embedding_out
    ).to(device)

    # load the model
    if model_type == 'AutoregressiveLSTM':
        model = encoder
    elif model_type == 'VAEAutoencoder':
        is_vae = params["is_vae"] if params is not None else False
        decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, num_layers=num_layers, is_decoder=True).to(device)
        model = VAEAutoencoder(encoder, decoder, hidden_size, is_vae, bottleneck_size).to(device)
    
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
        hidden_vecs = hidden_vecs.mean(dim=0)
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

    visualize_params_with_labels(pred_params, gt_params, string_labels, model_type)
    return mae.mean()


def visualize_params_with_labels(pred_params, gt_params, labels, model_type='AutoregressiveLSTM'):
    """Visualize the ground truth and predicted parameters."""
    # Ensure data is numpy array
    pred_params = np.array(pred_params)
    gt_params = np.array(gt_params)
    labels = np.array(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = cm.tab20(np.linspace(0, 1, len(labels) // 2)) # 2 sample trajectories per parameter set

    for i in range(len(gt_params)):
        ax1.scatter(gt_params[i, 0], gt_params[i, 1], color=colors[i//2])
        ax1.text(gt_params[i, 0], gt_params[i, 1], str(labels[i]), fontsize=8, ha='center')

        ax2.scatter(pred_params[i, 0], pred_params[i, 1], color=colors[i//2])
        ax2.text(pred_params[i, 0], pred_params[i, 1], str(labels[i]), fontsize=8, ha='center')

    ax1.set_title('Ground Truth Parameters')
    ax1.set_xlabel("m1")
    ax1.set_ylabel("m2")
    ax1.set_xlim([0.8, 2.2]), ax1.set_ylim([0.8, 2.2])
    # ax1.set_xlim([2.0, 3.0]), ax1.set_ylim([2.0, 3.0]) # evaluate generalizability to unseen parameters (extrapolation)

    ax2.set_title('Predicted Parameters')
    ax2.set_xlabel("m1")
    ax2.set_ylabel("m2")
    ax2.set_xlim([0.8, 2.2]), ax2.set_ylim([0.8, 2.2])
    # ax2.set_xlim([2.0, 3.0]), ax2.set_ylim([2.0, 3.0]) # evaluate generalizability to unseen parameters (extrapolation)

    plt.tight_layout()
    plt.savefig(f'./params_{model_type}.png')


def visualize_trajectory(ckpt_path="./ckpts/model_1000.pt", idx=0, model_type='AutoregressiveLSTM', params=None, visualize=True):
    """Analytically optimize a linear layer to map hidden vectors to parameters."""
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path="val_data.pickle", num_workers=4, shuffle=False)

    hidden_size = params["hidden_size"] if params is not None else 100
    predict_ahead = params["predict_ahead"] if params is not None else 1
    bottleneck_size = params["bottleneck_size"] if params is not None else -1
    num_layers = params["num_layers"] if params is not None else 1
    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        predict_ahead=predict_ahead,
        num_layers=num_layers
    ).to(device)

    # Load model
    if model_type == 'AutoregressiveLSTM':
        model = encoder
    elif model_type == 'VAEAutoencoder':
        is_vae = params["is_vae"]
        decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, num_layers=num_layers, is_decoder=True).to(device)
        model = VAEAutoencoder(encoder, decoder, hidden_size, is_vae, bottleneck_size).to(device)
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
        # Add first time step to predictions
        pred_traj = torch.concat((data[idx, 0, :].unsqueeze(0), predictions[idx]), axis=0)
        pred_traj = pred_traj.detach().cpu().numpy()
        gt_traj = data[idx, :, :].detach().cpu().numpy()
        gt_times = times[0, 0, :].detach().cpu().numpy()
        pred_times = gt_times
        mae = np.mean(np.abs(pred_traj - gt_traj))

    print("MAE (reconstruction) on the validation set: ", mae)
    
    if visualize:
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
    
    return mae


def visualize_pred_loss(ckpt_path="./ckpts/model_1000.pt", params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path="val_data_200_steps.pickle", num_workers=4, shuffle=False)

    hidden_size = params["hidden_size"] if params is not None else 100
    predict_ahead = params["predict_ahead"] if params is not None else 1
    is_vae = params["is_vae"] if params is not None else False
    bottleneck_size = params["bottleneck_size"] if params is not None else -1
    num_layers = params["num_layers"] if params is not None else 1
    
    encoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=predict_ahead, num_layers=num_layers).to(device)
    decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=199, num_layers=num_layers, is_decoder=True).to(device)
    model = VAEAutoencoder(encoder, decoder, hidden_size, is_vae, bottleneck_size).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    for i, (W, times, trajectories) in enumerate(dataloader):        
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape

        # Run model
        data = trajectories.view(-1, time_size, state_size)
        input = data[:, :99, :] # only first 100 timesteps are input to encoder
        targets = data[:, 1:, :]
        with torch.no_grad():
            predictions, _, _, _, _ = model(input, 199)
        
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mae_mean = np.mean(np.abs(predictions - targets), axis=0)
    mae_std = np.std(np.abs(predictions - targets), axis=0)
    
    loss_recon = np.mean(mae_mean[:99, :], axis=0)
    loss_pred = np.mean(mae_mean[99:, :], axis=0)

    # Plot MAE of placement
    _, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(mae_mean[:, 0], label='x1 (m1 displacement)')
    axes[0].fill_between(np.arange(199), mae_mean[:, 0] - mae_std[:, 0], mae_mean[:, 0] + mae_std[:, 0], alpha=0.2)
    
    axes[0].plot(mae_mean[:, 1], label='x2 (m2 displacement)')
    axes[0].fill_between(np.arange(199), mae_mean[:, 1] - mae_std[:, 1], mae_mean[:, 1] + mae_std[:, 1], alpha=0.2)
    
    # Annotation
    axes[0].axvline(x=100, color='grey', linestyle=':', linewidth=1)
    axes[0].annotate('Reconstruction', xy=(85, 4), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right')
    axes[0].annotate('Prediction', xy=(115, 4), arrowprops=dict(facecolor='black', shrink=0.05))

    axes[0].set_title("Reconstruction Loss: [x1] {:.2f}, [x2] {:.2f}\n".format(*loss_recon[:2]) + \
        "Prediction Loss: [x1] {:.2f}, [x2] {:.2f}".format(*loss_pred[:2])) 

    # Plot MAE of velocities
    axes[1].plot(mae_mean[:, 2], label='x1_dot (m1 velocity)')
    axes[1].fill_between(np.arange(199), mae_mean[:, 2] - mae_std[:, 2], mae_mean[:, 2] + mae_std[:, 2], alpha=0.2)

    axes[1].plot(mae_mean[:, 3], label='x2_dot (m2 velocity)')
    axes[1].fill_between(np.arange(199), mae_mean[:, 3] - mae_std[:, 3], mae_mean[:, 3] + mae_std[:, 3], alpha=0.2)
    
    # Annotation
    axes[1].axvline(x=100, color='grey', linestyle=':', linewidth=1)
    axes[1].annotate('Reconstruction', xy=(85, 5), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right')
    axes[1].annotate('Prediction', xy=(115, 5), arrowprops=dict(facecolor='black', shrink=0.05))
    
    axes[1].set_title("Reconstruction Loss: [x1_dot] {:.2f}, [x2_dot] {:.2f}\n".format(*loss_recon[2:]) + \
        "Prediction Loss: [x1_dot] {:.2f}, [x2_dot] {:.2f}".format(*loss_pred[2:])) 

    for i in range(2):
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Average Loss')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'./pred_loss')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generalizability():
    # Define parameters and dataset
    noise_amounts = [0, 0.01, 0.05, 0.1, 0.2]
    model_type = 'AutoregressiveLSTM'
    params = {"hidden_size": 100, "predict_ahead": 1, "bottleneck_size": -1, "num_layers": 4, "embedding_out": -1}

    # Dictionaries to store MAE for clean and noisy models
    mae_clean = {}
    mae_noisy = {}

    # Evaluate model trained with clean data
    model_path = "./ckpts/framework1_best_2000.pt"
    for noise in noise_amounts:
        train_data_path = f"train_data_{noise}noise.pickle"
        val_data_path = f"val_data_{noise}noise.pickle"
        mae_clean[noise] = evaluate(model_path, train_data_path, val_data_path, model_type, params)
    
    # Evaluate model trained with noisy data
    for noise in noise_amounts:
        train_data_path = f"train_data_{noise}noise.pickle"
        val_data_path = f"val_data_{noise}noise.pickle"
        if noise == 0:
            model_path = "./ckpts/framework1_best_2000.pt"
        else:
            model_path = f"./ckpts/model_2000_{noise}noise.pt"
        mae_noisy[noise] = evaluate(model_path, train_data_path, val_data_path, model_type, params)    

    # Prepare data for seaborn
    data = []
    for noise in noise_amounts:
        data.append({'Noise Level': noise*100, 'MAE': mae_clean[noise], 'Model trained on': 'clean dataset'})
        data.append({'Noise Level': noise*100, 'MAE': mae_noisy[noise], 'Model trained on': 'noisy dataset'})

    df = pd.DataFrame(data)

    # Set seaborn style
    sns.set(style="whitegrid")

    # Create seaborn bar plot
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Noise Level', y='MAE', hue='Model trained on', data=df, palette="muted")

    # Add labels and title
    bar_plot.set_xlabel('Noise Level (%)', fontsize=14)
    bar_plot.set_ylabel('MAE', fontsize=14)

    # Save and show plot
    plt.savefig(f'./generalizability.png')
    plt.show()


if __name__ == "__main__":
    # generalizability()

    # Evaluate parameters
    evaluate(
        "./ckpts/framework1_best_2000.pt", 
        "train_data.pickle",
        "val_data.pickle",
        'AutoregressiveLSTM', 
        {"hidden_size": 100, "predict_ahead": 1, "bottleneck_size": -1, "num_layers": 4, "embedding_out": -1}
    )
    # evaluate(
    #     "./ckpts/vae_model_2000.pt", 'VAEAutoencoder', 
    #     {"hidden_size": 100, "predict_ahead": 1, "is_vae": False, "bottleneck_size": 20, "num_layers": 8}
    # )
    
    # # # Trajectories
    # # visualize_trajectory("./ckpts/framework1_best.pt", 100, 'AutoregressiveLSTM', {"hidden_size": 100, "predict_ahead": 10})
    # visualize_trajectory(
    #     "./ckpts/vae_model_2000.pt", 0, 'VAEAutoencoder', 
    #      {"hidden_size": 100, "predict_ahead": 1, "is_vae": False, "bottleneck_size": 20, "num_layers": 8}
    # )
    
    # visualize_pred_loss(
    #     "./ckpts/vae_model_2000.pt",
    #     {"hidden_size": 100, "predict_ahead": 1, "is_vae": False, "bottleneck_size": 20, "num_layers": 8}
    # )