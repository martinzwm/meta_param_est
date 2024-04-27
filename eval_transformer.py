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


if __name__ == "__main__":
    # Create model
    config_file = "./configs/transformer_encoder_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    d_input = config['d_input']
    d_model = config['d_model']
    d_linear = config['d_linear']
    num_heads = config['num_heads']
    dropout = config['dropout']
    num_layers = config['num_layers']
    model = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers)
    # model = TransformerDecoder(d_input, d_model, d_linear, num_heads, dropout, num_layers)

    # Evaluate model
    evaluate_transformer(
        model,
        ckpt_path="./ckpts/encoder_best.pt", 
        train_data_path="./data/train_data.pickle",
        val_data_path="./data/val_data.pickle",
        log_result=True,
    )
