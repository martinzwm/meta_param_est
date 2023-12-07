import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamics import SpringMassSystem, get_dataloader
from eval import get_hidden_vectors_and_params, solve, evaluate
from model import AutoregressiveLSTM, VAEAutoencoder

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


def visualize_latent_gt(variable="m1"):
    """Generate a .gif of trajectories to visualize the effect of m1 and m2 on the system"""
    images = []
    for v in np.linspace(1, 2, 100):
        # Define dynamic system
        if variable == "m1":
            m1 = v
            m2 = 1.0
        elif variable == "m2":
            m1 = 1.0
            m2 = v
        W = torch.tensor([1.0, 1.0, m1, m2]) # k1, k2, m1, m2
        system = SpringMassSystem(W)

        # Define initial state
        initial_state = torch.tensor([1.0, -1.0, 0.0, 0.0])  # Initial displacements and velocities
        t_span = [0, 10]  # From t=0 to t=10
        dt = 0.1  # Time step

        # Generate trajectories
        times, trajectory = system.trajectory(initial_state, t_span, dt)

        # Save figure
        fig, ax = plt.subplots()
        ax.plot(times, trajectory[:, 0], label="x1 (m1 displacement)")
        ax.plot(times, trajectory[:, 1], label="x2 (m2 displacement)")
        ax.set_ylim([-3, 3])
        ax.set_xlabel("Time")
        ax.set_ylabel("Displacement")
        ax.legend()
        ax.grid(True)
        ax.set_title("k1 = {}, k2 = {}, m1 = {:.2f}, m2 = {:.2f}".format(*W.tolist()))
        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close()
    
    # Create a .gif of the trajectories
    import imageio
    imageio.mimsave(f'./trajectories_{variable}.gif', images, duration=100)


def get_control_knob(ckpt_path, model_type="VAEAutoencoder", params=None, variable="m1"):
    """Get the linear layer from the parameter space to the latent space."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=32, data_path="train_data.pickle", num_workers=8, shuffle=False)

    hidden_size = params["hidden_size"] if params is not None else 100
    predict_ahead = params["predict_ahead"] if params is not None else 1
    bottleneck_size = params["bottleneck_size"] if params is not None else -1
    encoder = AutoregressiveLSTM(
        hidden_size=hidden_size, 
        predict_ahead=predict_ahead
    ).to(device)

    # Load the model
    if model_type == 'AutoregressiveLSTM':
        model = encoder
    elif model_type == 'VAEAutoencoder':
        is_vae = params["is_vae"] if params is not None else False
        decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, is_decoder=True).to(device)
        model = VAEAutoencoder(encoder, decoder, hidden_size, is_vae, bottleneck_size).to(device)
          
    model.load_state_dict(torch.load(ckpt_path, map_location = device))
    model.eval()

    # Extract hidden vectors and parameters
    hidden_vecs, gt_params = get_hidden_vectors_and_params(model, dataloader, device, model_type) # hidden_vecs here has bias column
    hidden_vecs = hidden_vecs[:, :-1] # Remove bias column

    # Get corresponding latent vectors for (m1, m2) = (1, 1), (1, 2), (2, 1)
    X_1_1 = hidden_vecs[0:100].mean(dim=0)
    X_1_2 = hidden_vecs[400:500].mean(dim=0)
    X_2_1 = hidden_vecs[2000:2100].mean(dim=0)

    # # Solve for the linear system
    # linear_layer = solve(gt_params, hidden_vecs)

    if variable == "m1":
        return X_1_1, X_2_1 - X_1_1
    elif variable == "m2":
        return X_1_1, X_1_2 - X_1_1


def visualize_latent_pred(ckpt_path, model_type="VAEAutoencoder", params=None, variable="m1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the linear layer from the parameter space to the latent space
    X_1_1, linear_layer = get_control_knob(ckpt_path, model_type, params)
    X_1_1 = X_1_1.unsqueeze(0).to(device)
    linear_layer = linear_layer.unsqueeze(0).to(device)

    # Define dynamic system
    hidden_size = params["hidden_size"] if params is not None else 100
    encoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=1).to(device) # we won't use encoder
    decoder = AutoregressiveLSTM(hidden_size=hidden_size, predict_ahead=99, is_decoder=True).to(device)
    model = VAEAutoencoder(encoder, decoder, hidden_size, False, -1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location = device))

    images = []
    for v in np.linspace(1, 2, 100):
        # Get latent variable from parameter
        if variable == "m1":
            m1 = v
            m2 = 1.0
            latent = X_1_1 + (m1 - 1.0) * linear_layer
        elif variable == "m2":
            m1 = 1.0
            m2 = v
            latent = X_1_1 + (m2 - 1.0) * linear_layer

        latent = latent.unsqueeze(0)
        W = np.array([1.0, 1.0, m1, m2])

        # Define initial state
        initial_state = torch.tensor([[1, -1, 0.0, 0.0]]).to(device)  # Initial displacements and velocities
        initial_state = initial_state.unsqueeze(1).repeat(1, 99, 1)

        # Generate trajectories
        trajectory, _ = model.decoder.forward(initial_state, latent)
        trajectory = trajectory.squeeze().cpu().detach().numpy()
        times = np.arange(0.1, 10, 0.1)

        # Save figure
        fig, ax = plt.subplots()
        ax.plot(times, trajectory[:, 0], label="x1 (m1 displacement)")
        ax.plot(times, trajectory[:, 1], label="x2 (m2 displacement)")
        ax.set_ylim([-3, 3])
        ax.set_xlabel("Time")
        ax.set_ylabel("Displacement")
        ax.legend()
        ax.grid(True)
        ax.set_title("k1 = {}, k2 = {}, m1 = {:.2f}, m2 = {:.2f}".format(*W.tolist()))
        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close()
    
    # Create a .gif of the trajectories
    import imageio
    imageio.mimsave(f'./trajectories_{variable}_pred.gif', images, duration=100)




if __name__ == "__main__":
    visualize_latent_gt(variable="m1")
    visualize_latent_pred(
        ckpt_path="./ckpts/framework2_best_pred_loss.pt",
        model_type="VAEAutoencoder",
        params={"hidden_size": 100, "predict_ahead": 1, "is_vae": False, "bottleneck_size": -1},
        variable="m1"
    )

