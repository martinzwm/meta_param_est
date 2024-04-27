import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Code for a barebone transformer class"""
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_linear, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # Pointwise feedforward
        self.linear1 = nn.Linear(d_model, d_linear)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_linear, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            - x [=] (B, T, d)
            - mask [=] (T, T)
            - key_padding_mask [=] (N, T), used to prevent attention on special 
                tokens (e.g., <PAD>, <SEP>). A value of 1 means ignored, 0 means keep.
        """
        # Attention
        attn_output, alpha = self.multihead_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output) # dropout and residual connection
        x = self.norm1(x)
        
        # Feedforward
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(ff_output) # dropout and residual connection
        out = self.norm2(x)
        return out
     

"""Code for stacking multiple transformer block together"""
class Transformer(nn.Module):
    def __init__(self, d_input, d_model, d_linear, num_heads, dropout, num_layers, max_T=100):
        super(Transformer, self).__init__()
        # Define transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, d_linear, num_heads, dropout) for _ in range(num_layers)]
        )

        # Define non-transformer layers
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_linear = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.ReLU()
        )
        self.pos_E = nn.Embedding(max_T, d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        # Apply linear projection and add position embedding
        x = self.input_linear(x)
        B, T = x.shape[:2]
        pos_ids = torch.arange(T).expand(B, -1).to(x.device)
        x  = x + self.pos_E(pos_ids)

        # Concatenate [CLS] token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        return x

    def get_param(self, x):
        return self.head(x)


class TransformerDecoder(Transformer):

    def __init__(self, d_input, d_model, d_linear, num_heads, dropout, num_layers, max_T=100):
        super(TransformerDecoder, self).__init__(d_input, d_model, d_linear, num_heads, dropout, num_layers, max_T)
        self.output_linear = nn.Linear(d_model, d_input)

    def forward(self, x):
        # Apply linear projection and add position embedding
        x = self.input_linear(x)
        B, T = x.shape[:2]
        pos_ids = torch.arange(T).expand(B, -1).to(x.device)
        x  = x + self.pos_E(pos_ids)

        # Concatenate [CLS] token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Create masking
        mask = torch.triu(torch.ones(T+1, T+1)).to(x.device)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
    def pred_next_step(self, emb):
        return self.output_linear(emb)
    
    def generate(self, x, gen_T):
        with torch.no_grad():
            for i in range(gen_T):
                emb = self.forward(x)
                x_out = self.pred_next_step(emb[:, -1, :]).unsqueeze(1)
                x = torch.cat([x, x_out], dim=1)
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


    def test_transformer_encoder(self):
        # Define Transformer
        d_input = 4
        d_model = 64
        d_linear = 32
        num_heads = 8
        dropout = 0.1
        num_layers = 6
        transformer = Transformer(d_input, d_model, d_linear, num_heads, dropout, num_layers)
        
        # Define input
        input = torch.randn(16, 10, d_input)
        output = transformer(input)
        print(output.shape) # should be (16, 11, 64)

    
    def test_transformer_decoder(self):
         # Define Transformer
        d_input = 4
        d_model = 64
        d_linear = 32
        num_heads = 8
        dropout = 0.1
        num_layers = 6
        transformer = TransformerDecoder(d_input, d_model, d_linear, num_heads, dropout, num_layers)
        
        # Define input
        input = torch.randn(16, 10, d_input)
        output = transformer(input)
        print(output.shape) # should be (16, 11, 64)

        generated_traj = transformer.generate(input[:, 0:1, :], 10)
        print(generated_traj.shape) # should be (16, 11, 4), 11 = 1 + 10


if __name__ == "__main__":
    test_model = TestModel()
    # test_model.test_transformer_block()
    # test_model.test_transformer_encoder()
    test_model.test_transformer_decoder()