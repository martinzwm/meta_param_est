import pdb
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from autoencoder import Autoencoder, LinearAutoencoder, SimpleAutoencoder
from vorticity_dataset import get_image_dataloader

def eval_ae(model, ckpt_path, val_data_path, log_result=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Get data
    dataloader = get_image_dataloader(batch_size=32, data_path=val_data_path, num_workers=8, shuffle=False)
    mean, std = 0, 0.905
    
    total_loss = 0.0
    for i, image in enumerate(dataloader):
        # Get data
        image = image.to(device)
        image = (image - mean) / std

        # Forward pass
        out = model(image)
        loss = F.mse_loss(out, image)

        # Logging
        total_loss += loss.item()

    # Log
    if log_result:
        print("MSE on val set: {}".format(total_loss / len(dataloader)))
        # Reconstruction the first image
        image = image[0].cpu().detach().numpy()
        out = out[0].cpu().detach().numpy()
        image = (image * std) + mean
        out = (out * std) + mean

        # Comparison of original and reconstructed images
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(image[0])
        ax[0, 1].imshow(out[0])
        ax[1, 0].imshow(image[1])
        ax[1, 1].imshow(out[1])
        plt.savefig("reconstruction.png")

    return total_loss / len(dataloader)


if __name__ == "__main__":
    # Load configuration
    config_file = "./configs/ae_config.json"
    with open(config_file, "r") as f:
        default_config = json.load(f)
    dropout = default_config["dropout"]

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Autoencoder(dropout_rate=dropout).to(device)
    model = LinearAutoencoder(latent_dim=512).to(device)
    # model = SimpleAutoencoder(latent_dim=512, normalize=True).to(device)

    # Evaluate
    ckpt_path = "./ckpts/ae_linear_best.pt"
    val_data_path = "./data/vorticity_val.pickle"
    eval_ae(model, ckpt_path, val_data_path, log_result=True)
