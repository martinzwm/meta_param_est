import pdb
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch

from transformer import Transformer, TransformerDecoder
from dynamics import get_dataloader
from eval import visualize_params_with_labels

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


def get_embeddings(model, dataloader, device):
    """
    Collect model embeddings and gt system parameters.
    
    Args:
        - model: model to probe
        - dataloader: dataloader to run model on
    
    Returns:
        - X: embeddings
        - y: ground truth system parameters
    """
    model.eval()
    y, X = [], []
    for _, (W, _, trajectories) in enumerate(dataloader):
        W = W.to(device)       
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape
        data = trajectories.view(-1, time_size, state_size)

        # Run model - not sure why torch.no_grad() would lead to nan results, need to partition to fit in memory
        num_partition = 10
        data = torch.chunk(data, num_partition, dim=0)
        embedding_list = []
        for i in range(num_partition):
            embedding = model(data[i]).detach()
            embedding_list.append(embedding)
        embedding = torch.cat(embedding_list, dim=0)
        embedding = embedding[:, 0, :]

        W = W[:, 2:].repeat_interleave(sample_size, dim=0)

        X.append(embedding)
        y.append(W)
    
    # Add columns of ones for bias term
    X = torch.cat(X, dim=0)
    ones = torch.ones(X.shape[0], 1).to(device)
    X = torch.cat((X, ones), dim=1)

    y = torch.cat(y, dim=0)
    return X, y


def solve(X, Y):
    """
    Optimize a linear layer to map hidden vectors to parameters.
    """
    linear_layer = torch.linalg.lstsq(X, Y).solution
    return linear_layer


def evaluate_transformer(
    model,
    ckpt_path="./ckpts/model_1000.pt", 
    train_data_path="./data/train_data.pickle",
    val_data_path="./data/val_data.pickle",
    log_result=False,
):
    """Evaluate the model embeddings using linear probe to system parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Get linear probe layer from training data
    dataloader = get_dataloader(batch_size=32, data_path=train_data_path, num_workers=4, shuffle=False)
    x_train, y_train = get_embeddings(model, dataloader, device)
    linear_layer = solve(x_train, y_train)

    # Evaluate on training data
    pred_y_train = torch.mm(x_train, linear_layer)
    mae_train = torch.mean(torch.abs(pred_y_train - y_train), dim=0)
    mae_train = mae_train.cpu().numpy()

    # Evaluate on validation data
    dataloader = get_dataloader(batch_size=32, data_path=val_data_path, num_workers=4, shuffle=False)
    x_val, y_val = get_embeddings(model, dataloader, device)
    pred_y_val = torch.mm(x_val, linear_layer)
    mae_val = torch.mean(torch.abs(pred_y_val - y_val), dim=0)
    mae_val = mae_val.cpu().numpy()

    # Log results
    if log_result:
        print("MAE Train:", mae_train.mean())
        print("MAE Val:", mae_val.mean())
        # Visualization
        labels = [",".join(map(str, row)) for row in y_val.cpu().numpy()]
        visualize_params_with_labels(pred_y_val.cpu(), y_val.cpu(), labels, "Transformer")
    return mae_train, mae_val


def get_trajectories(model, dataloader, device, max_T):
    """
    Collect model predicted and gt trajectories.
    
    Args:
        - model: model to probe
        - dataloader: dataloader to run model on
    
    Returns:
        - pred_traj: predicted trajectories
        - gt_traj: ground truth trajectories
    """
    model.eval()
    pred_traj, gt_traj = [], []
    for _, (W, _, trajectories) in enumerate(dataloader):
        # Get gt trajectories
        W = W.to(device)       
        trajectories = trajectories.to(device)
        batch_size, sample_size, time_size, state_size = trajectories.shape
        data = trajectories.view(-1, time_size, state_size)
        gt_traj.append(data)

        # Get predicted trajectories
        # Method 1
        out = model.generate(data[:, 0:1, :], max_T).detach()

        # # Method 2
        # emb = model(data)
        # method2emb = emb
        # emb = emb[:, 1:-1, :]
        # out = model.pred_next_step(emb).detach()
        # out = torch.cat([data[:, 0:1, :], out], dim=1)
        # method2out = out

        # # Method 3
        # x = data[:, 0:1, :]
        # out = x
        # for i in range(max_T):
        #     emb = model(x)
        #     x_out = model.pred_next_step(emb[:, -1, :]).unsqueeze(1)
        #     out = torch.cat([out, x_out], dim=1).detach() # track predicted trajectory
        #     x = torch.cat([x, data[:, i+1:i+2, :]], dim=1) # use ground truth for next step

        pred_traj.append(out)
    
    # Concatenate
    pred_traj = torch.cat(pred_traj, dim=0)
    gt_traj = torch.cat(gt_traj, dim=0)
    return pred_traj, gt_traj


def visualize_trajectory(model, ckpt_path="./ckpts/model_1000.pt", val_data_path="./data/val_data.pickle", idx=0, visualize=True):
    """Analytically optimize a linear layer to map hidden vectors to parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

     # Load data
    dataloader = get_dataloader(batch_size=32, data_path=val_data_path, num_workers=4, shuffle=False)
    pred_traj, gt_traj = get_trajectories(model, dataloader, device, max_T=99)
    pred_traj = pred_traj.cpu().numpy()
    # pred_traj = np.concatenate([np.zeros((pred_traj.shape[0], 1, pred_traj.shape[2])), pred_traj], axis=1)
    gt_traj = gt_traj.cpu().numpy()

    mae = np.mean(np.abs(pred_traj - gt_traj))
    
    if visualize:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        times = np.arange(100)
        gt_traj = gt_traj[idx]
        pred_traj = pred_traj[idx]

        # Displacement subplot
        axs[0].plot(times, gt_traj[:, 0], label="x1 (m1 displacement)")
        axs[0].plot(times, gt_traj[:, 1], label="x2 (m2 displacement)")
        axs[0].plot(times, pred_traj[:, 0], label="x1 (pred)")
        axs[0].plot(times, pred_traj[:, 1], label="x2 (pred)")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Displacement")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_title("MAE = {:.4f}".format(mae))

        # Velocity subplot
        axs[1].plot(times, gt_traj[:, 2], label="x1_dot (m1 velocity)")
        axs[1].plot(times, gt_traj[:, 3], label="x2_dot (m2 velocity)")
        axs[1].plot(times, pred_traj[:, 2], label="x1_dot (pred)")
        axs[1].plot(times, pred_traj[:, 3], label="x2_dot (pred)")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Velocity")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'./traj_transformer.png')
    
    return mae


if __name__ == "__main__":
    # Create model
    config_file = "./configs/transformer_decoder_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    d_input = config['d_input']
    d_model = config['d_model']
    d_linear = config['d_linear']
    num_heads = config['num_heads']
    dropout = config['dropout']
    num_layers = config['num_layers']

    # model = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers)
    model = TransformerDecoder(d_input, d_model, d_linear, num_heads, dropout, num_layers)

    # # Evaluate model
    # evaluate_transformer(
    #     model,
    #     ckpt_path="./ckpts/decoder_contr_best.pt", 
    #     train_data_path="./data/train_data.pickle",
    #     val_data_path="./data/val_data.pickle",
    #     log_result=True,
    # )

    # Visualize trajectory
    visualize_trajectory(
        model,
        ckpt_path="./ckpts/model_2000.pt",
        val_data_path="./data/val_data.pickle",
        idx=0,
        visualize=True
    )