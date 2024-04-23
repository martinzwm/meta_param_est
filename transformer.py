import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Code for a barebone transformer class"""
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_linear, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        # Query, Key, Value projection matrices
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)

        # Pointwise feedforward
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
        # Query, Key, Value projection
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # Attention
        attn_output, _ = self.multihead_attn(q, k, v, mask)
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

        # Define non-transformer layers
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_linear = nn.Linear(d_input, d_model)

        # First transformer layer
        layers = [TransformerBlock(d_model, d_linear, num_heads, dropout)]

        # Rest of the transformer layers
        layers += [TransformerBlock(d_model, d_linear, num_heads, dropout) for _ in range(num_layers-1)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask=None):
        # Add class token and linear projection
        x = self.input_linear(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        return x


class TestModel:
     
    def test_transformer_block(self):
        # Define Transformer
        d_model = 64
        d_linear = 32
        num_heads = 8
        dropout = 0.1
        transformer = TransformerBlock(d_model, d_linear, num_heads, dropout)
        
        # Define input
        input = torch.randn(16, 10, d_model)
        output = transformer(input)
        print(output.shape)


    def test_transformer(self):
        # Define Transformer
        d_input = 4
        d_model = 64
        d_linear = 32
        num_heads = 8
        dropout = 0.1
        num_layers = 6
        transformer = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers)
        pdb.set_trace()
        
        # Define input
        input = torch.randn(16, 10, d_input)
        output = transformer(input)
        print(output.shape)


if __name__ == "__main__":
    test_model = TestModel()
    # test_model.test_transformer_block()
    test_model.test_transformer()