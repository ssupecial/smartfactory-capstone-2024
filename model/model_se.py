import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

from torch.autograd import Variable

import os
import sys
import copy
import math
import numpy as np
from typing import List
from matrix import extract_vec

FILE = "test.emb"


class Args:
    def __init__(self):
        self.lr = 0.001
        self.b1 = 0.9
        self.b2 = 0.999
        self.activation = "relu"
        self.epoch_num = 10
        self.final_layer_dim = 20


args = Args()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


from torch.autograd import Variable


class LearnedPositionEncoding1(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, input_size1=100, input_size2=20):
        super().__init__(d_model, input_size1 * input_size2)
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.view(
            self.d_model,
            self.input_size1,
            self.input_size2,
        ).unsqueeze(0)
        x = x + weight
        return self.dropout(x)


class LearnedPositionEncoding2(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, input_size=7):
        super().__init__(input_size * input_size, d_model)
        self.input_size = input_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.view(
            self.input_size * self.input_size, self.d_model
        ).unsqueeze(1)
        x = x + weight
        return self.dropout(x)


class MultiLevelTransformer(nn.Module):
    def __init__(
        self,
        args,
        input_dim: int = 64,
        patch_size: List[int] = [2, 2, 2],
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: List[int] = [4, 4, 4],
        num_decoder_layers: List[int] = [4, 4, 4],
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        final_layer_dim: int = 20,
    ):
        super(MultiLevelTransformer, self).__init__()
        # input dim은 차원의 수 (64)
        self.zeros = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=False)
        self.d_model = d_model
        self.patch_size = patch_size
        self.final_layer_dim = final_layer_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.global_position_embedding = LearnedPositionEncoding1(
            d_model=d_model, dropout=dropout
        )
        self.position_embedding = nn.ModuleList(
            [
                LearnedPositionEncoding2(
                    d_model=d_model, dropout=dropout, input_size=patch_size[i]
                )
                for i in range(len(patch_size))
            ]
        )

        encoder_norm = nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=args.activation,
                    ),
                    num_encoder_layers[i],
                    encoder_norm,
                )
                for i in range(len(patch_size))
            ]
        )

        self.bottle_neck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model // 4),
                    nn.LayerNorm(d_model // 4),
                    nn.GELU(),
                    nn.Linear(d_model // 4, d_model),
                    nn.LayerNorm(d_model),
                )
                for i in range(len(patch_size))
            ]
        )

        self.pre_conv = nn.Conv2d(input_dim, d_model, kernel_size=1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(672, 128), nn.ReLU(), nn.Linear(128, self.final_layer_dim)
        )
        self.final_layer = nn.Conv2d(
            self.final_layer_dim, self.final_layer_dim, kernel_size=1, bias=False
        )

    def calculate_size(self, level):
        S = self.patch_size[level]
        P = 1
        for i in range(level + 1, len(self.patch_size), 1):
            P *= self.patch_size[i]
        return P, S

    def forwardDOWN(self, x, encoder_block, position_embedding, level):
        _, BPSPS, C = x.size()
        P, S = self.calculate_size(level)
        B = BPSPS // (P * S * P * S)
        x = (
            x.view(B, P, S, P, S, C)
            .permute(2, 4, 0, 1, 3, 5)
            .contiguous()
            .view(S * S, B * P * P, C)
        )  # (SS, BPP, C)
        pad = self.zeros.expand(-1, B * P * P, -1)
        x = encoder_block(src=torch.cat((pad.detach(), position_embedding(x)), dim=0))

        latent_patch = x[0, :, :].unsqueeze(0).contiguous()  # (1, BPP, C)
        latent_pixel = x[1:, :, :].contiguous()  # (SS, BPP, C)
        # print(x.size())

        return latent_patch, latent_pixel

    def forward(self, x1, x2, x3):
        print("Initial shape:", x1.size())
        x1 = self.pre_conv(x1)
        x2 = self.pre_conv(x2)
        x3 = self.pre_conv(x3)

        B, C, H1, W1 = x1.size()  # (B, C, H, W) = (8, 512, 100, 20)
        B, C, H2, W2 = x2.size()  # (8, 512, 50, 10)
        B, C, H3, W3 = x3.size()  # (8, 512, 25, 5)
        print("After pre_conv:", x1.size())
        print("After pre_conv:", x2.size())
        print("After pre_conv:", x3.size())

        x1 = self.global_position_embedding(x1)
        print("After global_position_embedding:", x1.size())
        x1 = x1.permute(0, 2, 3, 1).contiguous().view(B * H1 * W1, C).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1).contiguous().view(B * H2 * W2, C).unsqueeze(0)
        x3 = x3.permute(0, 2, 3, 1).contiguous().view(B * H3 * W3, C).unsqueeze(0)
        print("After permute, view, unsqueeze:", x1.size())
        print("After permute, view, unsqueeze:", x2.size())
        print("After permute, view, unsqueeze:", x3.size())
        latent_list = []
        for i in range(len(self.encoder)):
            if i == 0:
                x, l = self.forwardDOWN(
                    x=x1,
                    encoder_block=self.encoder[i],
                    position_embedding=self.position_embedding[i],
                    level=i,
                )
            elif i == 1:
                x, l = self.forwardDOWN(
                    x=x2 + x,
                    encoder_block=self.encoder[i],
                    position_embedding=self.position_embedding[i],
                    level=i,
                )
            elif i == 2:
                x, l = self.forwardDOWN(
                    x=x3 + x,
                    encoder_block=self.encoder[i],
                    position_embedding=self.position_embedding[i],
                    level=i,
                )
            print(f"After forwardDOWN {i}: x size: {x.size()}, l size: {l.size()}")
            latent_list.append(self.bottle_neck[i](l))

        # Concatenate the latent representations along the channel dimension
        l_concat = torch.cat(
            latent_list, dim=1
        )  # Shape: [4, 250, 1536] concatenated along the feature dimension

        # Reshape l_concat to [B, H, W, -1] for MLP processing
        l_concat = l_concat.permute(1, 0, 2).contiguous().view(B * H1 * W1, -1)
        print("After concatenation:", l_concat.size())

        # Apply the MLP
        l_mlp = self.mlp(l_concat)
        print("After MLP:", l_mlp.size())

        # Reshape back to [B, H, W, 20] and then permute to [B, 20, H, W]
        l_mlp = (
            l_mlp.view(B, H1, W1, self.final_layer_dim).permute(0, 3, 1, 2).contiguous()
        )
        print("After final reshape:", l_mlp.size())

        return self.final_layer(l_mlp)


def Create_nets(args):
    transformer = MultiLevelTransformer(args)
    transformer.apply(weights_init_normal)
    return transformer


# Initialize the model
transformer = Create_nets(args)
matrix, _ = extract_vec(FILE)
# Create example input tensor
# example_input = torch.randn(8, 64, 100, 20)  # (B, C, H, W)
# example_input2 = torch.randn(8, 64, 50, 10)
# example_input3 = torch.randn(8, 64, 25, 5)


# Convert the matrix to a tensor and float type
matrix_tensor = torch.tensor(matrix).float()
# Reshape the matrix to (64, 100, 20)
reshaped_matrix = matrix_tensor.view(64, 100, 20)
# Replicate the reshaped matrix 8 times to form (8, 64, 100, 20)
example_input = reshaped_matrix.unsqueeze(0).repeat(8, 1, 1, 1)
# Generate random indices for rows and columns for example_input2
row_indices_2 = torch.randperm(100)[:50]
col_indices_2 = torch.randperm(20)[:10]

# Select the random rows and columns for example_input2
example_input2 = example_input[:, :, row_indices_2, :][:, :, :, col_indices_2]

# Generate random indices for rows and columns for example_input3
row_indices_3 = torch.randperm(50)[:25]
col_indices_3 = torch.randperm(10)[:5]

# Select the random rows and columns for example_input3 from example_input2
example_input3 = example_input2[:, :, row_indices_3, :][:, :, :, col_indices_3]

# Pass the input tensor through the model
output1 = transformer(example_input, example_input2, example_input3)

# Print the shapes of the outputs
print("Output1 shape:", output1.shape)
# 8, 20, 100, 20 (B, C, H, W)
