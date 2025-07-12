import torch
import torch.nn as nn

class MLP_L63(nn.Module):
    def __init__(self, d, latent_dim):
        super(MLP_L63, self).__init__()
        self.d = d
        self.latent_dim = latent_dim

        self.layer1 = nn.Linear(2 * d, latent_dim)
        self.layer2 = nn.Linear(latent_dim, latent_dim)
        # self.layer3 = nn.Linear(latent_dim, latent_dim)
        self.layer4 = nn.Linear(latent_dim, d)

        self.relu = nn.ReLU()

    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         if m == self.layer4:
    #             nn.init.zeros_(m.weight)
    #             nn.init.zeros_(m.bias)
    #         else:
    #             nn.init.normal_(m.weight, mean=0, std=1e-4)
    #             nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        # out = self.relu(self.layer3(out))
        out = self.layer4(out)
        return out

class Simple_MLP(nn.Module):
    def __init__(self, d_input, d_output, latent_dim=64, num_hidden_layers=2):
        super(Simple_MLP, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(d_input, latent_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(latent_dim, d_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class Upsampling_MLP(nn.Module):
    def __init__(self, d_input, d_output, latent_dim=64, num_hidden_layers=2):
        super(Upsampling_MLP, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(d_input, latent_dim))
        layers.append(nn.ReLU())

        temp = latent_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(temp, 2*temp))
            layers.append(nn.ReLU())
            temp *= 2

        layers.append(nn.Linear(temp, d_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class NaiveNetwork(nn.Module):
    def __init__(self, d):
        super(NaiveNetwork, self).__init__()
        self.d = d

    def forward(self, v):
        return torch.zeros(v.shape[0], self.d).to(v.device)
    
class AttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_attention_layers=1, hidden_dim=64):
        super(AttentionModel, self).__init__()
        # Input linear layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Variable number of attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
            for _ in range(num_attention_layers)
        ])
        # Additional fully connected layers
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc3.bias, 0)
        # Initialize weights for attention layers
        for attention in self.attention_layers:
            nn.init.normal_(attention.in_proj_weight, mean=0.0, std=1e-3)
            nn.init.constant_(attention.in_proj_bias, 0)
            nn.init.normal_(attention.out_proj.weight, mean=0.0, std=1e-3)
            nn.init.constant_(attention.out_proj.bias, 0)

    def forward(self, x):
        # Apply the first linear layer and activation
        x = self.relu(self.fc1(x))
        # Prepare for attention layers (requires sequence dimension)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, hidden_dim]
        # Apply each attention layer
        for attention in self.attention_layers:
            x, _ = attention(x, x, x)
        # Remove the sequence dimension
        x = x.squeeze(0)  # Shape: [batch_size, hidden_dim]
        # Apply the remaining fully connected layers
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ComplexAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexAttentionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        
        self.attention1 = nn.MultiheadAttention(embed_dim=32, num_heads=2)
        self.attention2 = nn.MultiheadAttention(embed_dim=32, num_heads=2)

        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(32)
    
    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.fc4.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.fc5.weight, mean=0.0, std=1e-3)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.fc4.bias, 0)
        nn.init.constant_(self.fc5.bias, 0)

        nn.init.normal_(self.attention1.in_proj_weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.attention1.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.attention2.in_proj_weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.attention2.out_proj.weight, mean=0.0, std=1e-3)

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.dropout(x)  
        x = self.relu(self.fc3(x))  

        x = x.unsqueeze(0)
        x, _ = self.attention1(x, x, x)
        x = self.norm(x)
        x, _ = self.attention2(x, x, x)
        x = self.norm(x)
        x = x.squeeze(0)

        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# class MAB(nn.Module):
#     """
#     Multihead Attention Block (MAB)
#     """
#     def __init__(self, dim_Q, dim_KV, num_heads):
#         super(MAB, self).__init__()
#         self.dim_V = dim_Q  # Output dimension matches the query dimension
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=self.dim_V, num_heads=num_heads)
#         self.ln1 = nn.LayerNorm(self.dim_V)
#         self.ln2 = nn.LayerNorm(self.dim_V)
#         self.ffn = nn.Sequential(
#             nn.Linear(self.dim_V, self.dim_V),
#             nn.ReLU(),
#             nn.Linear(self.dim_V, self.dim_V)
#         )

#     def forward(self, Q, K):
#         # Q, K: [batch_size, N, dim]
#         Q_norm = self.ln1(Q)
#         K_norm = self.ln1(K)
#         Q_norm = Q_norm.transpose(0, 1)  # Convert to [N, batch_size, dim]
#         K_norm = K_norm.transpose(0, 1)
#         attn_output, _ = self.multihead_attn(Q_norm, K_norm, K_norm)
#         attn_output = attn_output.transpose(0, 1)  # Convert back to [batch_size, N, dim]
#         H = Q + attn_output  # Residual connection
#         H_norm = self.ln2(H)
#         H = H + self.ffn(H_norm)  # Residual connection
#         return H

class MAB(nn.Module):
    """
    Multihead Attention Block (MAB)
    """
    def __init__(self, dim_Q, dim_KV, num_heads, freeze_WQ=False):
        super(MAB, self).__init__()
        self.dim_V = dim_Q  # Output dimension matches the query dimension
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.dim_V, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(self.dim_V)
        self.ln2 = nn.LayerNorm(self.dim_V)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim_V, self.dim_V),
            nn.ReLU(),
            nn.Linear(self.dim_V, self.dim_V)
        )
        self.freeze_WQ = freeze_WQ
        if self.freeze_WQ:
            self._freeze_WQ()

    def _freeze_WQ(self):
        # Get the in_proj_weight parameter
        in_proj_weight = self.multihead_attn.in_proj_weight
        embed_dim = self.multihead_attn.embed_dim
        # Set W_Q to the identity matrix
        with torch.no_grad():
            in_proj_weight[:embed_dim, :] = torch.eye(embed_dim)
        # Register a backward hook to zero out gradients for W_Q
        def hook(grad):
            grad[:embed_dim, :] = 0
            return grad
        in_proj_weight.register_hook(hook)
        # Handle the bias term if it exists
        if self.multihead_attn.in_proj_bias is not None:
            in_proj_bias = self.multihead_attn.in_proj_bias
            with torch.no_grad():
                in_proj_bias[:embed_dim] = 0
            # Register a backward hook for the bias
            def bias_hook(grad):
                grad[:embed_dim] = 0
                return grad
            in_proj_bias.register_hook(bias_hook)

    def forward(self, Q, K):
        # Q, K: [batch_size, N, dim_V]
        Q_norm = self.ln1(Q)
        K_norm = self.ln1(K)
        Q_norm = Q_norm.transpose(0, 1)  # Convert to [N, batch_size, dim_V]
        K_norm = K_norm.transpose(0, 1)
        attn_output, _ = self.multihead_attn(Q_norm, K_norm, K_norm)
        attn_output = attn_output.transpose(0, 1)  # Convert back to [batch_size, N, dim_V]
        H = Q + attn_output  # Residual connection
        H_norm = self.ln2(H)
        H = H + self.ffn(H_norm)  # Residual connection
        return H

class SAB(nn.Module):
    """
    Self-Attention Block (SAB)
    """
    def __init__(self, dim_in, dim_out, num_heads):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, num_heads)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, X):
        H = self.mab(X, X)
        return self.fc(H)

class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA)
    """
    def __init__(self, dim, num_heads, num_seeds, freeze_WQ=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))  # Seed vectors
        self.mab = MAB(dim, dim, num_heads, freeze_WQ)

    def forward(self, X):
        batch_size = X.size(0)
        S = self.S.repeat(batch_size, 1, 1)  # [batch_size, num_seeds, dim]
        return self.mab(S, X)

class SetTransformer(nn.Module):
    """
    Set Transformer main model with customizable number of SAB layers.
    """
    def __init__(self, input_dim, num_heads, num_inds, output_dim, hidden_dim, num_layers=2, freeze_WQ=False):
        super(SetTransformer, self).__init__()
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Self-attention encoder blocks
        self.enc = nn.Sequential(
            *[SAB(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)]
        )
        # Pooling layer
        self.pma = PMA(hidden_dim, num_heads, num_inds, freeze_WQ)
        # Self-attention decoder blocks
        self.dec = nn.Sequential(
            *[SAB(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)]
        )
        # Output layer
        self.fc_out = nn.Linear(hidden_dim * num_inds, output_dim)
    
    def forward(self, X):
        # X: [batch_size, N, input_dim]
        H = self.embedding(X)
        H = self.enc(H)
        H = self.pma(H)
        H = self.dec(H)
        H = H.view(H.size(0), -1)  # Flatten
        output = self.fc_out(H)
        return output

class TransformerBlock(nn.Module):
    """
    Standard Transformer Block without positional encoding.
    """
    def __init__(self, dim, num_heads, dim_feedforward):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(p=0.1)  # Adjusted dropout
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # Adjusted eps value
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)  # Adjusted eps value
        self.dropout1 = nn.Dropout(p=0.1)  # Adjusted dropout
        self.dropout2 = nn.Dropout(p=0.1)  # Adjusted dropout
        self.activation = nn.GELU()  # ReLU replaced with GELU

    def forward(self, src):
        # src shape: [N, batch_size, dim]
        src2, _ = self.self_attn(src, src, src)
        src = src + 0.1 * self.dropout1(src2)  # Scaled residual connection
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + 0.1 * self.dropout2(src2)  # Scaled residual connection
        src = self.norm2(src)
        return src

class EquivariantTransformer(nn.Module):
    """
    Transformer model that maps input tensors of shape [batch_size, N, D_1] to output tensors of shape [batch_size, N, D_2].
    """
    def __init__(self, input_dim, output_dim, num_heads, num_layers, dim_feedforward, hidden_dim):
        super(EquivariantTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # X: [batch_size, N, input_dim]
        H = self.embedding(X)  # [batch_size, N, hidden_dim]
        H = H.transpose(0, 1)  # Transpose to [N, batch_size, hidden_dim]
        for layer in self.layers:
            H = layer(H)
        H = H.transpose(0, 1)  # Back to [batch_size, N, hidden_dim]
        output = self.fc_out(H)  # [batch_size, N, output_dim]
        return output
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model with the specified parameters
    model = EquivariantTransformer(
        input_dim=80,
        output_dim=40,
        num_heads=8,
        num_layers=16,
        dim_feedforward=64,
        hidden_dim=64
    ).to(device)

    # Define a simple test input and target
    batch_size = 4
    N = 10  # Number of elements in the set
    input_dim = 80
    output_dim = 40

    # Generate random input tensor
    X = torch.randn(batch_size, N, input_dim, requires_grad=True).to(device)

    # Generate random target tensor
    target = torch.randn(batch_size, N, output_dim).to(device)

    # Define a simple loss function
    criterion = nn.MSELoss()

    # Forward pass
    output = model(X)

    # Compute loss
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    # Function to check gradients
    def check_gradients(model):
        gradients_exist = True
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Parameter '{name}' has no gradient!")
                gradients_exist = False
            else:
                grad_norm = param.grad.data.norm()
                print(f"Gradient norm for parameter '{name}': {grad_norm}")
        if gradients_exist:
            print("\nAll parameters have gradients.")
        else:
            print("\nSome parameters do not have gradients!")

    # Check gradients
    check_gradients(model)
