import pdb
import numpy as np

import torch
import torch.nn as nn

from dynamics import *


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


class TrajectoryLSTM(nn.Module):
    """LSTM model to predict the trajectory of a spring-mass system."""
    def __init__(self, input_size=4, hidden_size=10, num_layers=1, output_size=4):
        super(TrajectoryLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        self.linear = nn.Linear(hidden_size, output_size)
        self.linear_param = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        # Passing the input through the LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        final_output = self.linear(lstm_out)
        return final_output, c_n
    
    def get_embedding(self, c_n):
        # Get hidden state embedding
        h = self.linear_param(c_n)
        return h
    

class AutoregressiveLSTM(nn.Module):
    """Autoregressive LSTM model to predict the trajectory of a spring-mass system."""
    def __init__(self, input_size=4, hidden_size=10, num_layers=1, output_size=4, predict_ahead=5):
        super(AutoregressiveLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_ahead = predict_ahead
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        outputs = []

        # Initialize hidden and cell states
        h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Process input sequence
        for t in range(seq_len-self.predict_ahead+1):
            lstm_out, (h_t, c_t) = self.lstm(x[:, t:t+1, :], (h_t, c_t))
            (h_t_auto, c_t_auto) = (h_t, c_t)
            lstm_out = self.linear(lstm_out)
            output_t = [lstm_out]
    
            # Autoregressive prediction for the next 5 steps
            for _ in range(self.predict_ahead-1):  # We already have the output for the first future timestep
                lstm_out, (h_t_auto, c_t_auto) = self.lstm(lstm_out, (h_t_auto, c_t_auto))
                lstm_out = self.linear(lstm_out)
                output_t.append(lstm_out)
            outputs.append(torch.concat(output_t, dim=1))
    
        # Concatenate outputs
        final_output = torch.stack(outputs, dim=1)
        return final_output, c_t


class VAEAutoencoder(nn.Module):
    """VAE Autoencoder for reconstructing time series with a variational latent space and optional contrastive loss capability."""
    def __init__(self, encoder, decoder, z_dims=10):
        super(VAEAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.mu = nn.Linear(encoder.hidden_size, z_dims)
        self.logvar = nn.Linear(encoder.hidden_size, z_dims)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        _, seq_len, _ = x.shape
        
        # Encoding
        _, c_t = self.encoder(x)
        mu = self.mu(c_t)
        logvar = self.logvar(c_t)

        # Reparameterization
        z = self.reparameterize(mu, logvar)
        z = z.permute(1, 0, 2) # batch_size, seq_len, hidden_size
        z = z.repeat(1, seq_len, 1) # repeat the timestep channel to seq_len
        
        # Decoding
        recon_x, _ = self.decoder(z)
        return recon_x.squeeze(1), mu, logvar, c_t


class TestModel:

    def __init__(self):
        pass

    def test_lstm(self):
        test_dynamics = TestDynamics()
        # Get trajectory
        times, trajectory = test_dynamics.test_dynamics()

        # Define LSTM
        lstm = TrajectoryLSTM()

        # Define input and target
        input = trajectory[:-1, :].unsqueeze(0)
        target = trajectory[1:, :]
        print(input.shape)
        print(target.shape)

        # Forward pass
        output = lstm.get_embedding(input)
        print(output.shape)
    
    def test_autoregressive_lstm(self):
        lstm = AutoregressiveLSTM()
        input_tensor = torch.randn(16, 10, 4)  # Example input of shape (B, T, C)
        output, c_t = lstm(input_tensor)
        print(output.shape)  # Should be (B, T * 10, C)
        print(c_t.shape)
        
    def test_vae_autoencoder(self):
        input_size = 4
        hidden_size = 10
        output_size = 4
        z_dims = 4
        predict_ahead = 10
        seq_len = 20 # hard code

        encoder = AutoregressiveLSTM(input_size, hidden_size, 1, output_size, predict_ahead)
        decoder = AutoregressiveLSTM(z_dims, hidden_size, 1, output_size, seq_len)
        autoencoder = VAEAutoencoder(encoder, decoder, z_dims)
        
        input_tensor = torch.randn(16, seq_len, input_size)
        recon_x, mu, logvar, c_t = autoencoder(input_tensor)  
        print('recon_x: ', recon_x.shape)
        print('mu: ', mu.shape)
        print('logvar: ', logvar.shape) 
        print('c_t: ', c_t.shape) 
        assert recon_x.shape == input_tensor.shape


if __name__ == "__main__":
    unittests = TestModel()
    # unittests.test_lstm()
    # unittests.test_autoregressive_lstm()
    unittests.test_vae_autoencoder()
