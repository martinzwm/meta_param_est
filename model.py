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
    def __init__(self, input_size=4, hidden_size=10, embedding_out=-1, num_layers=1, output_size=4, predict_ahead=5, is_decoder=False):
        super(AutoregressiveLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        self.linear = nn.Linear(hidden_size, output_size)
        if embedding_out > 0.0:
            self.linear_param = nn.Linear(hidden_size, embedding_out)
        else:
            self.linear_param = nn.Linear(hidden_size, hidden_size)
            
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_ahead = predict_ahead
        self.is_decoder = is_decoder
    
    def forward(self, x, c_0=None):
        batch_size, seq_len, _ = x.shape
        outputs = []

        # Initialize hidden and cell states
        h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_t = c_0 if self.is_decoder else torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_t = c_t.to(x.device)

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

    def get_embedding(self, c_n):
        # Get hidden state embedding
        latent = self.linear_param(c_n)
        return latent


class VAEAutoencoder(nn.Module):
    """VAE Autoencoder for reconstructing time series with a variational latent space and optional contrastive loss capability."""
    def __init__(self, encoder, decoder, hidden_size=10, is_vae=True, bottleneck_size=-1):
        super(VAEAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.mu = nn.Linear(encoder.hidden_size, hidden_size)
        self.logvar = nn.Linear(encoder.hidden_size, hidden_size)
        self.is_vae = is_vae

        self.bottleneck_size = bottleneck_size
        if self.bottleneck_size > 0.0:
            self.linear_down = nn.Linear(hidden_size, bottleneck_size)
            self.linear_up = nn.Linear(bottleneck_size, hidden_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, out_len=None):
        _, seq_len, _ = x.shape
        
        # Encoding
        encoder_out, c_t = self.encoder(x)
        c_t = self.encoder.get_embedding(c_t)
        
        # Reparameterization
        mu = self.mu(c_t)
        logvar = self.logvar(c_t)
        if self.is_vae:
            z = self.reparameterize(mu, logvar)
        else:
            z = c_t

        # Bottleneck
        bottleneck = None
        if self.bottleneck_size > 0.0:
            z = self.linear_down(z)
            bottleneck = z
            z = self.linear_up(z)
        
        # Decoding
        if out_len is not None:
            seq_len = out_len
        x = x[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1) # use the first timestep of x, i.e. x_0, as input to decoder
        recon_x, _ = self.decoder(x, z)

        if self.bottleneck_size > 0.0:
            return recon_x.squeeze(1), encoder_out, mu, logvar, bottleneck
        else:
            return recon_x.squeeze(1), encoder_out, mu, logvar, z
        

import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_input, d_model):
        super(Attention, self).__init__()
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)

    def forward(self, x, mask=None):
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        out, attn = self.scaled_dot_product_attention(query, key, value, mask)
        return out, attn

    def scaled_dot_product_attention(self, query, key, value, mask):
        """
        Args:
            - query [=] (B, T, d)
            - key [=] (B, T, d)
            - value [=] (B, T, d)
            - mask [=] (T, T). Element to be masked are filled with 0.
        Returns:
            - out [=] (B, T, d)
            - attn: attention matrix of probabilities [=] (B, T)
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        scores /= np.sqrt(query.size(-1))

        if mask:
            scores = scores.masked_fill(mask==0, -1e-9)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, value)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_input, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.heads.append(Attention(d_input, d_model))
        for _ in range(num_heads-1):
            self.heads.append(Attention(d_model, d_model))
        self.linear = nn.Linear(num_heads * d_model, d_model)
    
    def forward(self, x, attn_mask=None):
        outs = []
        for head in self.heads:
            out, attn_score = head(x, attn_mask)
            outs.append(out)

        outs = torch.cat(outs, dim=1)
        outs = self.linear(outs)
        return outs


class TransformerBlock(nn.Module):
    def __init__(self, d_input, d_model, d_linear, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.multihead_attn = MultiHeadAttention(d_input, d_model, num_heads) # TODO: use nn.MultiheadAttention
        self.linear1 = nn.Linear(d_model, d_linear)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_linear, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, key_padding_mask=None):
        """
        Args:
            - x [=] (B, T, d)
            - mask [=] (T, T)
            - key_padding_mask [=] (N, T), used to prevent attention on special 
                tokens (e.g., <PAD>, <SEP>). A value of 1 means ignored, 0 means keep.
        """
        # Attention
        # attn_output, _ = self.multihead_attn(x, x, x, 
        #                                     attn_mask=mask, key_padding_mask=key_padding_mask)
        attn_output, _ = self.multihead_attn(x, mask)
        x = x + self.dropout(attn_output) # dropout and residual connection
        x = self.norm1(x)
        
        # Feedforward
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(ff_output) # dropout and residual connection
        out = self.norm2(x)
        return out
     

"""Code for stacking multiple transformer block together"""
class Transformer(nn.Module):
    def __init__(self, d_input, d_model, d_linear, num_heads, dropout, num_layers):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_input, d_model, d_linear, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_input)) # TODO: should it be d_model?

    def forward(self, x, mask=None):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


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
        output, hidden = lstm(input)
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
        num_layers = 1
        predict_ahead = 10
        seq_len = 99

        encoder = AutoregressiveLSTM(input_size, hidden_size, -1, num_layers, output_size, predict_ahead, False)
        decoder = AutoregressiveLSTM(input_size, hidden_size, -1, num_layers, output_size, seq_len, True)
        autoencoder = VAEAutoencoder(encoder, decoder, hidden_size)
        
        input_tensor = torch.randn(16, seq_len, input_size)
        recon_x, encoder_out, mu, logvar, c_t = autoencoder(input_tensor)  
        print('recon_x: ', recon_x.shape)
        print('encoder_out: ', encoder_out.shape)
        print('mu: ', mu.shape)
        print('logvar: ', logvar.shape) 
        print('c_t: ', c_t.shape) 
        assert recon_x.shape == input_tensor.shape

    def test_transformer(self):
        # Define Transformer
        d_input = 4
        d_model = 32
        d_linear = 32
        num_heads = 4
        dropout = 0.1
        num_layers = 6
        transformer = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers)
        
        # Define input
        input = torch.randn(16, 10, d_input)
        output = transformer(input)
        print(output.shape)


if __name__ == "__main__":
    unittests = TestModel()
    # unittests.test_lstm()
    # unittests.test_autoregressive_lstm()
    # unittests.test_vae_autoencoder()
    unittests.test_transformer()
