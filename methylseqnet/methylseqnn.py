import torch
import torch.nn as nn

from methylseqnet.layers import *

batch_size = 64
num_epochs = 100
input_channels = 5
seq_length = 896
output_channels = 1

class MethylSeqNN(nn.Module):
    def __init__(self, input_channels, seq_length, output_channels):
        super(BassetCpG, self).__init__()

        # Trunk
        self.conv_dna = ConvDNA(input_channels, 288, 17, 3)
        self.conv_tower = ConvTower(288, 288, 512, 16, 5, 2, 6)
        self.conv_block = ConvBlock(508, 256, 1)
        self.dense_block = DenseBlock(1536, 768, 0.2)

        # Head
        self.final = Final(768, output_channels)

    def forward(self, x):
        x = self.conv_dna(x)
        x = self.conv_tower(x)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense_block(x)
        x = self.final(x)
        x = x.unsqueeze(1)
        return x