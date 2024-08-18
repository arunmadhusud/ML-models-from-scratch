import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        """
        Args:
            img_size (int): Size of the input image (assuming square image).
            patch_size (int): Size of each patch (assuming square patches).
            in_chans (int): Number of input channels (e.g., 3 for RGB images).
            embed_dim (int): Dimension of the embedding space.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolutional layer to project each patch to the embedding dimension
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_chans, img_size, img_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches, embed_dim).
        """
        # Apply the convolutional layer to project patches to the embedding dimension
        x = self.proj(x)  # [batch_size, embed_dim, n_patches**0.5, n_patches**0.5]
        
        # Flatten the spatial dimensions (height and width) into one dimension
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        
        # Transpose the tensor to get the shape [batch_size, n_patches, embed_dim]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0, proj_p=0):
        """
        Initialize the Attention module.

        Parameters:
        - dim (int): Dimensionality of input features and output features.
        - n_heads (int, optional): Number of attention heads. Default is 12.
        - qkv_bias (bool, optional): Whether to include bias in the query, key, value linear transformations. Default is True.
        - attn_p (float, optional): Dropout probability applied to the attention scores. Default is 0 (no dropout).
        - proj_p (float, optional): Dropout probability applied to the output of the linear projection layer. Default is 0 (no dropout).
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Linear transformations for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)

        # Linear transformation for output projection
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        """
        Perform the forward pass through the Attention module.

        Parameters:
        - x (torch.Tensor): Input tensor with shape [batch_size, n_patches+1, dim].

        Returns:
        - torch.Tensor: Output tensor after applying attention mechanism and projection with shape [batch_size, n_patches+1, dim].
        """
        # Linear transformation for query, key, value
        qkv = self.qkv(x)  # [batch_size, n_patches+1, dim*3]

        # Reshape and permute dimensions to split into query, key, value for each head
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Separate into q, k, v for each head

        # Compute scaled dot-product attention scores
        attn = (q @ k.transpose(2, 3)) * self.scale  # [batch_size, n_heads, n_patches+1, n_patches+1]

        # Apply softmax along the last dimension (n_patches+1) to normalize attention scores
        attn = nn.Softmax(dim=-1)(attn)

        # Apply dropout to attention scores
        attn = self.attn_drop(attn)

        # Weighted average of values based on attention scores
        weighted_avg = attn @ v  # [batch_size, n_heads, n_patches+1, head_dim]

        # Transpose dimensions to bring head_dim before n_heads
        weighted_avg = weighted_avg.transpose(1, 2)  # [batch_size, n_patches+1, n_heads, head_dim]

        # Flatten last two dimensions (n_heads, head_dim) into a single dimension
        weighted_avg = weighted_avg.flatten(2)  # [batch_size, n_patches+1, dim]

        # Linear transformation for final projection
        x = self.proj(weighted_avg)

        # Apply dropout to the output of the projection
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0):
        """
        Multi-Layer Perceptron (MLP) module.

        Parameters:
        - in_features (int): Number of input features.
        - hidden_features (int): Number of hidden units in the MLP.
        - out_features (int): Number of output features.
        - p (float, optional): Dropout probability. Default is 0 (no dropout).
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
        """
        Forward pass of the MLP.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, n_patches + 1, in_features).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, n_patches + 1, out_features).
        """
        x = self.fc1(x)        # Linear transformation (batch_size, n_patches + 1, hidden_features)
        x = self.act(x)        # GELU activation (batch_size, n_patches + 1, hidden_features)
        x = self.drop(x)       # Dropout (batch_size, n_patches + 1, hidden_features)
        x = self.fc2(x)        # Linear transformation (batch_size, n_patches + 1, out_features)
        x = self.drop(x)       # Dropout (batch_size, n_patches + 1, out_features)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=True, p=0, attn_p=0):
        """
        Transformer block module.

        Parameters:
        - dim (int): Dimensionality of input and output tensors.
        - n_heads (int): Number of attention heads.
        - mlp_ratio (int, optional): Ratio of hidden units to input dimensionality for MLP. Default is 4.
        - qkv_bias (bool, optional): Whether to include bias in the query, key, value linear transformations. Default is True.
        - p (float, optional): Dropout probability for MLP. Default is 0 (no dropout).
        - attn_p (float, optional): Dropout probability for attention scores. Default is 0 (no dropout).
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=self.hidden_features, out_features=dim, p=p)

    def forward(self, x):
        """
        Forward pass of the Transformer block.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, ..., dim).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, ..., dim).
        """
        x = x + self.attn(self.norm1(x))   # Residual connection with attention mechanism
        x = x + self.mlp(self.norm2(x))    # Residual connection with MLP
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4, qkv_bias=True, p=0, attn_p=0):
        """
        Vision Transformer (ViT) model.

        Parameters:
        - img_size (int, optional): Size of input image (both height and width). Default is 384.
        - patch_size (int, optional): Size of each image patch. Default is 16.
        - in_chans (int, optional): Number of input channels (e.g., 3 for RGB). Default is 3.
        - n_classes (int, optional): Number of classes for classification. Default is 1000.
        - embed_dim (int, optional): Dimensionality of the token embeddings. Default is 768.
        - depth (int, optional): Number of transformer blocks. Default is 12.
        - n_heads (int, optional): Number of attention heads in each transformer block. Default is 12.
        - mlp_ratio (int, optional): Ratio of hidden units to embed_dim in MLP. Default is 4.
        - qkv_bias (bool, optional): Whether to include bias in the query, key, value linear transformations. Default is True.
        - p (float, optional): Dropout probability. Default is 0 (no dropout).
        - attn_p (float, optional): Dropout probability for attention scores. Default is 0 (no dropout).
        """
        super().__init__()

        # Patch Embedding layer
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))

        # Dropout layer for positional embeddings
        self.pos_drop = nn.Dropout(p=p)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p) for _ in range(depth)])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_chans, img_size, img_size).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, n_classes).
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # Expand and concatenate class token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, 1 + n_patches, embed_dim)

        # Add positional embeddings and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Forward through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply layer normalization
        x = self.norm(x)  # (batch_size, 1 + n_patches, embed_dim)

        # Output classification logits for the class token
        cls_token_final = x[:, 0]  # Extract just the class token (batch_size, embed_dim)
        x = self.head(cls_token_final)  # (batch_size, n_classes)

        return x











        