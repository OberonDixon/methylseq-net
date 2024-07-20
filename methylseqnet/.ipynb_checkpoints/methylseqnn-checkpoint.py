import torch
import torch.nn as nn

from methylseqnet.layers import *

class MethylSeqNN(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_tracks,
    ):
        super(MethylSeqNN, self).__init__()

        # Trunk
        self.conv_dna = ConvDNA(in_channels=in_channels)
        self.conv_tower = ConvTower()
        self.conv_block = ConvBlock()
        self.conv_dropout = ConvDropout()
        # self.dense_block = DenseBlock(**hyperparams["DenseBlock"])

        # Head
        self.conv_final = ConvFinal(filters=out_tracks)
        # self.final = Final(**hyperparams["Final"])

    def forward(self, x):
        x = self.conv_dna(x)
        x = self.conv_tower(x)
        x = self.conv_block(x)
        x = self.conv_dropout(x)
        x = self.conv_final(x)

        
        # x = x.view(x.size(0), -1)  # Flatten
        # x = self.dense_block(x)
        # x = self.final(x)
        # x = x.unsqueeze(1)
        return x