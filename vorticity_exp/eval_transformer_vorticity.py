import pdb
import json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './../')
from model.transformer import Transformer, TransformerDecoder, EncoderDecoder
from eval_transformer import solve

from autoencoder import LinearAutoencoder, DummyAutoencoder
from vorticity_dataset import get_test_dataloader


def get_embeddings(model, autoencoder, dataloader, device, num_pt_enc):
    """
    Collect model embeddings and gt system parameters.
    
    Args:
        - model: model to probe
        - dataloader: dataloader to run model on
    
    Returns:
        - X: embeddings
        - y: ground truth system parameters
    """
    mean, std = 0, 0.905

    model.eval()
    y, X = [], []
    for _, (trajectories, vorticity, _) in enumerate(dataloader):
        # Calculate embedding
        trajectories = trajectories.to(device)
        B, T, C, H, W = trajectories.shape
        trajectories = (trajectories - mean) / std
        data = trajectories.view(B, T, -1)

        # Run model
        with torch.no_grad():
            features = autoencoder.encoder(data[:, :num_pt_enc, :])
        enc_emb = model.encoder(features)
        X.append(enc_emb[:, 0, :].detach())
        y.append(vorticity)
    
    # Add columns of ones for bias term
    X = torch.cat(X, dim=0)
    ones = torch.ones(X.shape[0], 1).to(device)
    X = torch.cat((X, ones), dim=1)
    
    y = torch.cat(y, dim=0)
    X = X.cpu().float()
    y = y.cpu().float()
    return X, y


def eval_embeddings(
    model, 
    autoencoder,
    model_ckpt_path,
    autoencoder_ckpt_path,
    train_data_path,
    val_data_path, 
    log_result
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))

    # Load autoencoder
    autoencoder.to(device)
    autoencoder.eval()
    autoencoder.load_state_dict(torch.load(autoencoder_ckpt_path, map_location=device))
    
    # Load data
    train_dataloader = get_test_dataloader(batch_size=32, data_path=train_data_path, num_workers=8, shuffle=True)
    val_dataloader = get_test_dataloader(batch_size=32, data_path=val_data_path, num_workers=8)
    
    # Get embeddings
    x_train, y_train = get_embeddings(model, autoencoder, train_dataloader, device, num_pt_enc=20)
    linear_layer = solve(x_train, y_train)
    linear_layer = linear_layer.reshape(-1, 1)

    # Evaluate on training data
    pred_y_train = torch.mm(x_train, linear_layer).squeeze()
    mae_train = torch.mean(torch.abs(pred_y_train - y_train), dim=0)
    mae_train = mae_train.cpu().numpy()

    # Evaluate on validation data
    x_val, y_val = get_embeddings(model, autoencoder, val_dataloader, device, num_pt_enc=20)
    pred_y_val = torch.mm(x_val, linear_layer).squeeze()
    mae_val = torch.mean(torch.abs(pred_y_val - y_val), dim=0)
    mae_val = mae_val.cpu().numpy()

    # Log results
    if log_result:
        print("MAE Train:", mae_train.mean())
        print("MAE Val:", mae_val.mean())
        # Visualization
        plt.figure()
        plt.plot(y_val.cpu().numpy(), pred_y_val.cpu().numpy(), '.')
        plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c="black")
        plt.xlabel("GT Vorticity")
        plt.ylabel("Predicted Voriticity")
        plt.savefig("vorticity_pred.png")
    return mae_train, mae_val


def get_trajectories(model, autoencoder, dataloader, device, max_T, num_pt_enc):
    """
    Collect model predicted and gt trajectories.
    
    Args:
        - model: model to probe
        - dataloader: dataloader to run model on
    
    Returns:
        - pred_traj: predicted trajectories
        - gt_traj: ground truth trajectories
    """
    mean, std = 0, 0.905

    model.eval()
    pred_traj, gt_traj = [], []
    for i, (trajectories, _, _) in tqdm.tqdm(enumerate(dataloader)):
        # Get gt trajectories
        trajectories = trajectories.to(device)
        gt_traj.append(trajectories[:, num_pt_enc:, :, :, :])
        B, T, C, H, W = trajectories.shape
        trajectories = (trajectories - mean) / std

        # Autoencoder
        data = trajectories.view(B, T, -1)
        with torch.no_grad():
            features = autoencoder.encoder(data[:, :num_pt_enc, :])

        # Transfomer
        with torch.no_grad():
            enc_emb = model.encoder(features)
            out = model.decoder.generate(features[:, -5:, :], max_T-num_pt_enc+1, enc_emb).detach()
            out = out[:, 5:, :]
            with torch.no_grad():
                out = autoencoder.decoder(out)
            out = out.view(B, -1, C, H, W)
            pred_traj.append(out)
        if i > 50:
            break

    # Concatenate
    pred_traj = torch.cat(pred_traj, dim=0)
    pred_traj = (pred_traj * std) + mean
    gt_traj = torch.cat(gt_traj, dim=0)
    return pred_traj, gt_traj


def eval_transformer(
    model, 
    autoencoder,
    model_ckpt_path,
    autoencoder_ckpt_path,
    val_data_path, 
    log_result,
    idx=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))

    # Load autoencoder
    autoencoder.to(device)
    autoencoder.eval()
    autoencoder.load_state_dict(torch.load(autoencoder_ckpt_path, map_location=device))

    # Load data
    val_dataloader = get_test_dataloader(batch_size=32, data_path=val_data_path, num_workers=8, shuffle=False)
    
    # Get trajectories
    pred_traj, gt_traj = get_trajectories(model, autoencoder, val_dataloader, device, max_T=39, num_pt_enc=20)
    pred_traj = pred_traj.detach().cpu().numpy()
    gt_traj = gt_traj.detach().cpu().numpy()

    mse = np.mean((pred_traj - gt_traj) ** 2)

    if log_result:
        print(f"MSE (traj): {mse}")
        
        # Visualization
        fig, ax = plt.subplots(4, 10, figsize=(20, 10))
        for j in range(10):
            ax[0, j].imshow(gt_traj[idx, j, 0, :, :])
            ax[1, j].imshow(gt_traj[idx, j, 1, :, :])
            ax[2, j].imshow(pred_traj[idx, j, 0, :, :])
            ax[3, j].imshow(pred_traj[idx, j, 1, :, :])
            ax[0, j].axis("off")
            ax[1, j].axis("off")
            ax[2, j].axis("off")
            ax[3, j].axis("off")

        fig.savefig("traj_pred_{}.png".format(idx))
    return mse


if __name__ == "__main__":
    # Load config
    with open("./configs/transformer_config.json", "r") as f:
        config = json.load(f)
    d_input = config['d_input']
    d_model = config['d_model']
    d_linear = config['d_linear']
    num_heads = config['num_heads']
    dropout = config['dropout']
    num_layers = config['num_layers']

    # Load model
    encoder = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers)
    decoder = TransformerDecoder(d_input, d_model, d_linear, num_heads, dropout, num_layers)
    model = EncoderDecoder(encoder, decoder)
    autoencoder = LinearAutoencoder(latent_dim=d_input)
    # autoencoder = DummyAutoencoder()


    # Evaluate
    mse = eval_transformer(
        model, 
        autoencoder,
        './ckpts/transformer_5000.pt',
        './ckpts/ae_linear_best.pt',
        './data/vorticity_train.pickle', 
        True,
        idx=236, # best: 1787, worst: 236
    )
    
    # mse_param_train, mse_param_val = eval_embeddings(
    #     model, 
    #     autoencoder,
    #     './ckpts/transformer_best.pt',
    #     './ckpts/ae_linear_best.pt',
    #     './data/vorticity_train.pickle',
    #     './data/vorticity_val.pickle', 
    #     True,
    # )